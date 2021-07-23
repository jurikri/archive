  ix0 = np.where(Y[:,1]==0)[0]
    ix1 = np.where(Y[:,1]==1)[0]
    
    class_mu_sigma = msFunction.msarray([2,5]) # classnum, fnnum
    for fnum in range(5):
        plt.figure()
        samples = X[ix0, fnum]
        x, y = gaus(mu = np.mean(samples), sigma = np.std(samples)); plt.plot(x,y)
        class_mu_sigma[0][fnum] = [np.mean(samples),np.std(samples)]
        
        samples = X[ix1, fnum]
        x, y = gaus(mu = np.mean(samples), sigma = np.std(samples)); plt.plot(x,y)
        class_mu_sigma[1][fnum] = [np.mean(samples),np.std(samples)]
        
        
    estimate = np.zeros(len(X)) * np.nan
    for n in range(len(X)):
        # class 0일 확률
        cl = 0
        prior = np.mean(Y[:,cl]==1)
        updates = prior
        for fnum in [1,0,3]:
            mu = class_mu_sigma[cl][fnum][0]
            sigma = class_mu_sigma[cl][fnum][1]
            p = stats.norm.pdf(X[n,fnum], mu, sigma)
            updates = updates * p
        pcl0 = updates
        
        cl = 1
        prior = np.mean(Y[:,cl]==1)
        updates = prior
        for fnum in [1,0,3]:
            mu = class_mu_sigma[cl][fnum][0]
            sigma = class_mu_sigma[cl][fnum][1]
            p = stats.norm.pdf(X[n,fnum], mu, sigma)
            updates = updates * p
        pcl1 = updates
        
        if pcl0 > pcl1: estimate[n] = 0
        if pcl0 < pcl1: estimate[n] = 1
        
    1 - np.mean(np.abs(estimate - Y[:,1]))
            
