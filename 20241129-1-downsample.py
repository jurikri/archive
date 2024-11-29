def downsample(baseline_power):  
    time_tlen = baseline_power.shape[2]
    baseline_power_down = []
    dt = 5
    for t3 in range(0, time_tlen, dt):
        baseline_power_down.append(np.median(baseline_power[:,:,t3:t3+dt], axis=2))
    baseline_power_down = np.array(baseline_power_down)
    return baseline_power_down
