import torch
import torch.nn.functional as F
import numpy as np
from models import *
from tool.dataloader import *
from seed import set_seed
from sklearn.metrics import roc_auc_score, f1_score
from scipy.sparse.csgraph import connected_components
import copy
import matplotlib.pyplot as plt
import networkx as nx

online_dataset=get_online_01()

def get_top_connected_components(edge_index, num_nodes, top_k=5):
    adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
    adj[edge_index[0], edge_index[1]] = 1
    adj[edge_index[1], edge_index[0]] = 1
    adj_np = adj.cpu().numpy()
    n_components, labels = connected_components(adj_np, directed=False)
    

    component_sizes = {}
    for i in range(n_components):
        size = (labels == i).sum()
        component_sizes[i] = size
    

    sorted_components = sorted(component_sizes.items(), key=lambda x: x[1], reverse=True)
    top_components = sorted_components[:top_k]
    

    selected_nodes = []
    for comp_id, _ in top_components:
        node_indices = torch.where(torch.tensor(labels == comp_id, device=edge_index.device))[0]
        selected_nodes.extend(node_indices.tolist())
    

    return selected_nodes, labels, top_components

def build_complementary_graph(edge_index, selected_nodes, num_nodes):


    selected_mask = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
    selected_mask[selected_nodes] = True
    

    selected_edges_mask = selected_mask[edge_index[0]] & selected_mask[edge_index[1]]
    complementary_edges = edge_index[:, selected_edges_mask]
    

    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_nodes)}
    new_edges = torch.stack([
        torch.tensor([old_to_new[int(complementary_edges[0, i])] for i in range(complementary_edges.shape[1])], 
                     device=edge_index.device),
        torch.tensor([old_to_new[int(complementary_edges[1, i])] for i in range(complementary_edges.shape[1])], 
                     device=edge_index.device)
    ])
    
    return new_edges, selected_nodes, old_to_new

def iterate_offline_model_with_fusion(x, xedge, y_reduced, optimizer):

    student_pretrain_model.train()
    student_classifier.train()
    optimizer.zero_grad()
    
    y_for_pretrain = (y_reduced >= 0).bool()
    data_en, dx, attention_weights, selected_mask = student_pretrain_model(
        x, xedge, y_for_pretrain, use_fusion=False
    )
    
    out = student_classifier(data_en, xedge)
    labeled_mask = (y_reduced >= 0)
    if labeled_mask.sum() > 0:
        labeled_out = out[labeled_mask]
        labeled_y = y_reduced[labeled_mask]
        classification_loss = F.cross_entropy(labeled_out, labeled_y.long())
    else:
        classification_loss = torch.tensor(0.0, device=x.device)
    
    node_mask = adaptive_feature_masking(xedge, x, mask_ratio=0.2)
    x_masked = x.clone()
    x_masked[node_mask] = 0
    _, data_en_masked, _, _ = student_pretrain_model(x_masked, xedge, use_fusion=True)
    
    if torch.isnan(data_en_masked).any() or torch.isinf(data_en_masked).any():
        attr_mask_loss = torch.tensor(0.0, device=x.device)
    else:
        attr_recon = torch.clamp(data_en_masked, min=-1e3, max=1e3)
        attr_mask_loss = F.mse_loss(attr_recon[node_mask], x[node_mask])
    
    alpha = 0.1
    beta = 10
    loss = (classification_loss + attr_mask_loss)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student_pretrain_model.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(student_classifier.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()

def iterate_offline_model_with_complement(x, xedge, y_reduced, optimizer):

    student_pretrain_model.train()
    student_classifier.train()
    optimizer.zero_grad()
    
    y_for_pretrain = (y_reduced >= 0).bool()
    data_en, dx, attention_weights, selected_mask = student_pretrain_model(
        x, xedge, y_for_pretrain, use_fusion=True
    )
    
    out = student_classifier(data_en, xedge)
    labeled_mask = (y_reduced >= 0)
    if labeled_mask.sum() > 0:
        labeled_out = out[labeled_mask]
        labeled_y = y_reduced[labeled_mask]
        classification_loss = F.cross_entropy(labeled_out, labeled_y.long())
    else:
        classification_loss = torch.tensor(0.0, device=x.device)

    attention_reg_loss = torch.tensor(0.0, device=x.device)
    if attention_weights is not None and selected_mask is not None:
        selected_attention = attention_weights[selected_mask].mean()
        unselected_attention = attention_weights[~selected_mask].mean()
        attention_reg_loss = F.relu(unselected_attention - selected_attention)

    alpha = 0.1
    beta = 10
    loss = (classification_loss + beta * attention_reg_loss)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student_pretrain_model.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(student_classifier.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()

def fiedler_vector_lobpcg(laplacian, tol=1e-6):
    n = laplacian.size(0)
    eigenvalues, eigenvectors = torch.lobpcg(
        laplacian, 
        k=2,  
        largest=False,  
        tol=tol
    )
    return eigenvectors[:, 1]

def compute_feature_importance_from_laplacian(edge_index, node_features, epsilon=1e-3):
    num_nodes = node_features.size(0)
    device = node_features.device
    
    node_features = (node_features - node_features.mean(dim=0)) / (node_features.std(dim=0) + 1e-6)
    
    adj = torch.zeros(num_nodes, num_nodes, device=device)
    adj[edge_index[0], edge_index[1]] = 1
    adj[edge_index[1], edge_index[0]] = 1
    
    adj_np = adj.cpu().numpy()
    n_components, labels = connected_components(adj_np, directed=False)
    
    epsilon = min(1e-3 * (n_components / num_nodes), 1e-2)
    
    adj_perturbed = adj + epsilon * (torch.ones(num_nodes, num_nodes, device=device) - torch.eye(num_nodes, device=device))
    
    degree_perturbed = adj_perturbed.sum(dim=1)
    degree_matrix_perturbed = torch.diag(degree_perturbed)
    laplacian = degree_matrix_perturbed - adj_perturbed
    fiedler_vector = fiedler_vector_lobpcg(laplacian)
    fiedler_vector = fiedler_vector / (torch.norm(fiedler_vector) + 1e-6)
    
    feature_importance = torch.abs(torch.matmul(node_features.t(), fiedler_vector))
    
    return feature_importance

def adaptive_feature_masking(edge_index, node_features, mask_ratio=0.2):
    num_nodes = node_features.size(0)
    feature_dim = node_features.size(1)
    device = node_features.device
    
    adj = torch.zeros(num_nodes, num_nodes, device=device)
    adj[edge_index[0], edge_index[1]] = 1
    adj[edge_index[1], edge_index[0]] = 1
    adj_np = adj.cpu().numpy()
    n_components, labels = connected_components(adj_np, directed=False)
    
    if n_components > num_nodes * 0.8:
        return torch.rand(num_nodes, feature_dim, device=device) < mask_ratio
    
    feature_importance = compute_feature_importance_from_laplacian(edge_index, node_features)
    
    feature_importance = (feature_importance - feature_importance.min()) / \
                         (feature_importance.max() - feature_importance.min() + 1e-6)
    
    mask_prob = feature_importance * mask_ratio * 2
    mask_prob = torch.clamp(mask_prob, 0.15, 0.5) 
    
    if torch.isnan(mask_prob).any() or torch.isinf(mask_prob).any():
        print("Mask probabilities contain NaN or Inf")
        return torch.rand(num_nodes, feature_dim, device=device) < mask_ratio
    
    feature_mask = torch.rand(feature_dim, device=device) < mask_prob
    node_mask = feature_mask.unsqueeze(0).expand(num_nodes, -1)

    return node_mask

def inter_class_loss(centroid1, centroid2, pos_features, neg_features, t=2.0, eps=1e-6):
    combined_features = torch.cat([pos_features, neg_features], dim=0)
    cov_matrix = torch.cov(combined_features.T)
    cov_matrix = cov_matrix + eps * torch.eye(cov_matrix.size(0), device=cov_matrix.device)
    cov_inv = torch.inverse(cov_matrix)
    diff = centroid1 - centroid2
    mahalanobis_dist = torch.sqrt(diff.T @ cov_inv @ diff)
    L_inter = - (t * mahalanobis_dist**2) / 2
    return L_inter
    
def train_offline_model(epoch, pretrain_date, optimizer):
    pretrain_data = []
    for j in range(epoch):
        loss_sum = 0
        for i in range(pretrain_date):
            data_now = torch.Tensor(online_dataset[i][0]).to(device)
            edge_now = torch.LongTensor(online_dataset[i][1]).long().to(device)
            y_reduced_now = torch.Tensor(online_dataset[i][3]).long().to(device)  
            pretrain_data.append([data_now, edge_now])
            loss = iterate_offline_model_with_fusion(data_now, edge_now, y_reduced_now, optimizer)
            loss_sum += loss
        print("Offline Epoch:", j, "Loss:", loss_sum)

def train_offline_model_next(epoch, pretrain_date, optimizer):
    pretrain_data = []
    for j in range(epoch):
        loss_sum = 0
        for i in range(pretrain_date):
            data_now = torch.Tensor(online_dataset[i][0]).to(device)
            edge_now = torch.LongTensor(online_dataset[i][1]).long().to(device)
            y_reduced_now = torch.Tensor(online_dataset[i][3]).long().to(device)  
            pretrain_data.append([data_now, edge_now])
            loss = iterate_offline_model_with_complement(data_now, edge_now, y_reduced_now, optimizer)
            loss_sum += loss
        print("Offline Epoch:", j, "Loss:", loss_sum)

#---------------------------------------MeanTeacher--------------------------------------------------------------------------------------------------
class MeanTeacher:
    def __init__(self, student_model, teacher_model, ema_decay=0.999):
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.ema_decay = ema_decay
        
    def update_teacher(self):
        for student_param, teacher_param in zip(self.student_model['encoder'].parameters(), 
                                               self.teacher_model['encoder'].parameters()):
            teacher_param.data = self.ema_decay * teacher_param.data + (1 - self.ema_decay) * student_param.data
        for student_param, teacher_param in zip(self.student_model['classifier'].parameters(), 
                                               self.teacher_model['classifier'].parameters()):
            teacher_param.data = self.ema_decay * teacher_param.data + (1 - self.ema_decay) * student_param.data

    def get_teacher_predictions(self, x, edge_index):
        self.teacher_model['encoder'].eval()
        self.teacher_model['classifier'].eval()
        with torch.no_grad():
            data_en = self.teacher_model['encoder'].get_encoder_output(x, edge_index, use_fusion=True)
            out = self.teacher_model['classifier'](data_en, edge_index)
            out = F.softmax(out, dim=1)
            _, pred = out.max(dim=1)
            confidence = out.max(dim=1)[0]
        return pred, confidence, data_en

def iterate_online_model_with_mean_teacher(x, xedge, teacher_pred, teacher_confidence, optimizer):
    student_pretrain_model.train()
    student_classifier.train()
    optimizer.zero_grad()
    
    data_en, _, attention_weights, selected_mask = student_pretrain_model(x, xedge, use_fusion=True)
    out = student_classifier(data_en, xedge)
    
    with torch.no_grad():
        teacher_result = mean_teacher.teacher_model['encoder'](x, xedge, use_fusion=True)
        if isinstance(teacher_result, tuple) and len(teacher_result) >= 2:
            teacher_data_en = teacher_result[0]  
        else:
            teacher_data_en = teacher_result
        teacher_out = mean_teacher.teacher_model['classifier'](teacher_data_en, xedge)
        teacher_out = F.softmax(teacher_out, dim=1) 

    similarity_loss = F.kl_div(
        F.log_softmax(out, dim=1),  
        teacher_out,                 
        reduction='batchmean'
    )

   

    attention_reg_loss = torch.tensor(0.0, device=x.device)
    if attention_weights is not None and selected_mask is not None:

        selected_attention = attention_weights[selected_mask].mean()
        unselected_attention = attention_weights[~selected_mask].mean()
        attention_reg_loss = F.relu(unselected_attention - selected_attention)


    similarity_weight = 10 
    beta = 1  
    total_loss = (
        similarity_loss * similarity_weight + beta * attention_reg_loss
    )
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()

aucu_all = []
f1u_all = []
accu_all = []
seed = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


set_seed(seed)
    
student_pretrain_model = EnhancedADDer(32, 32).to(device)
student_classifier = Classifier(32, 2).to(device)
teacher_pretrain_model = copy.deepcopy(student_pretrain_model).to(device)
teacher_classifier = copy.deepcopy(student_classifier).to(device)

optimizer_pretrain = torch.optim.Adam(student_pretrain_model.parameters(), lr=0.003)
train_optimizer = torch.optim.Adam([
        {'params': student_pretrain_model.parameters(), 'lr': 0.001},
        {'params': student_classifier.parameters(), 'lr': 0.003}
    ], lr=0.01)

mean_teacher = MeanTeacher(
        student_model={'encoder': student_pretrain_model, 'classifier': student_classifier},
        teacher_model={'encoder': teacher_pretrain_model, 'classifier': teacher_classifier},
        ema_decay=0.99
    )

pretrain_days = 14
pretrain_data = train_offline_model(100, pretrain_days, train_optimizer)
pretrain_data = train_offline_model_next(100, pretrain_days, train_optimizer)

teacher_pretrain_model.load_state_dict(student_pretrain_model.state_dict())
teacher_classifier.load_state_dict(student_classifier.state_dict())

num = 0
aucu = []
f1u = []
accu = []

print("Online Training")

for i in range(pretrain_days, len(online_dataset)):
    teacher_pred, teacher_confidence, teacher_features = mean_teacher.get_teacher_predictions(
            torch.Tensor(online_dataset[i][0]).to(device),
            torch.LongTensor(online_dataset[i][1]).to(device)
        )
        
    student_pretrain_model.eval()
    student_classifier.eval()
    data_now = torch.Tensor(online_dataset[i][0]).to(device)
    edge_now = torch.LongTensor(online_dataset[i][1]).to(device)
    y_now = torch.Tensor(online_dataset[i][2]).bool().to(device)

    data_en, _, _, _ = student_pretrain_model(data_now, edge_now, use_fusion=True)
    out = student_classifier(data_en, edge_now)
    out = F.softmax(out, dim=1)
    _, pred = out.max(dim=1)
       
    correct = pred.eq(y_now).sum().item()
    acc = correct / (len(y_now))
    accu.append(acc)
    class_probs = out[:, 1].detach().cpu().numpy()
    auc = roc_auc_score(y_now.cpu(), class_probs)
    aucu.append(auc)

    preds_np = pred.cpu().detach().numpy()
    y_true_np = y_now.cpu().detach().numpy()

    f1 = f1_score(y_true_np, preds_np, labels=[1], average='binary')
    f1u.append(f1)
        
    if num % 2 == 1:
        for numbers in range(10):
            iterate_online_model_with_mean_teacher(data_now, edge_now, teacher_pred, teacher_confidence, train_optimizer)
        mean_teacher.update_teacher()
    
    num += 1
        
print("Online AUC:", np.mean(aucu))
print("Online F1:", np.mean(f1u))
print("Online ACC:", np.mean(accu))
