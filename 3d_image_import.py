median_t4 = np.zeros((N,5)) * np.nan

import msfilepath
import skimage.io as skio


for SE in pslGroup + shamGroup:
    try:
        path, behav_data, raw_filepath, endsw = msfilepath.msfilepath1(SE)
        for se in range(3):
            file_path = path + '\\00' + str(se+1) + '.tif'
            imstack1 = skio.imread(file_path, plugin="tifffile")
            msplot = np.mean(np.mean(np.array(imstack1), axis=1), axis=1)
        #    plt.figure()
        #    plt.plot(msplot)
        #    plt.ylim([0, np.nanmedian(msplot) * 3])
            
            print(se, np.nanmedian(msplot))
            median_t4[SE,se] = np.nanmedian(msplot)
    except: pass
