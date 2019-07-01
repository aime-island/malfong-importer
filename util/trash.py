""" from imblearn.under_sampling import RandomUnderSampler

# Stilla ratio af noisy audio
if (args.noise_ratio):
    # Beyta random under sampler á values
    rus = RandomUnderSampler(sampling_strategy=args.noise_ratio, random_state=args.random_state)
    data_res, _ = rus.fit_resample(data.values, data['environment'].values)

    # Breyta gögnum aftur í dataframe
    data = pd.DataFrame(data_res, columns=column_names) """