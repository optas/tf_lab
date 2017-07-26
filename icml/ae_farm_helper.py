def relevant_class(loss):
    if loss == 'chamfer':
        return Chamfer_Stats
    elif loss == 'emd':
        return EMD_Stats
    else:
        return None


class Chamfer_Stats():
    experiments_ids = [4, 5, 6, 7, 8]

    experiment_id_to_bneck = {4: 64, 5: 256, 6: 512, 7: 32, 8: 128}

    best_epochs = {32: (980, 0.00023145349890876625),
                   64: (970, 0.00020686166095647683),
                   128: (970, 0.00020017094667833303),
                   256: (1000, 0.0002053646348071166),
                   512: (1000, 0.00019155694878900324)}

    # This is epoch where each architecture reached the training loss achieved by the worse performing architecture (i.e., for the one with the 32-bottleneck layer).
    max_min_epochs = {32: (980, 0.000231453498909),
                      64: (470, 0.000232941098663),
                      128: (380, 0.000232541767805),
                      256: (470, 0.000232944812497),
                      512: (350, 0.000231615319654)}


class EMD_Stats():
    experiments_ids = [10, 11, 12, 13, 14, 15, 16, 17]

    experiment_id_to_bneck = {10: 32, 11: 64, 12: 128, 13: 256, 14: 512, 15: 1024, 16: 16, 17: 8}

    best_epochs = {32: (960, 56.104974105668369),
                   64: (940, 54.250119469024398),
                   128: (990, 51.128213850366365),
                   256: (980, 51.021433926219608),
                   512: (990, 50.992049003639345),
                   1024: (950, 50.998874370522046),
                   16: (960, 58.259854856723528)
                   }

    # This is epoch where each architecture reached the training loss achieved by the worse performing architecture (i.e., for the one with the 32-bottleneck layer).
    max_min_epochs = {32: (960, 56.1049741057),
                      64: (590, 56.1521346463),
                      128: (290, 56.2572510551),
                      256: (290, 56.3530345713),
                      512: (260, 56.4351911322)}
