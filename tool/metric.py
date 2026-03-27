def get_avg(source):
    averages = []
    total = 0
    count = 0
    
    for value in source:
        total += value
        count += 1

        if count % 30 == 0:
            averages.append(total / count)
            total = 0
            count = 0

    if count > 0:
        averages.append(total / count)
    
    return averages