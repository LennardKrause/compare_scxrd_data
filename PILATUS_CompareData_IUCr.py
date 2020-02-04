import numpy as np
import glob, argparse, os
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt

class DefaultHelpParser(argparse.ArgumentParser):
    def error(self, message):
        import sys
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

def read_sfrm(fname):
    import re
    import numpy as np
    #def chunkstring(string, length):
    #    '''
    #     return header as list of tuples
    #      - splits once at ':'
    #      - keys and values are stripped strings
    #      - values with more than 1 entry are un-splitted
    #    '''
    #    return list(tuple(map(lambda i: i.strip(), string[0+i:length+i].split(':', 1))) for i in range(0, len(string), length)) 
    #header_list = chunkstring(header, 80)
    with open(fname, 'rb') as f:
        # read the first 512 bytes
        # find keyword 'HDRBLKS' 
        header_0 = f.read(512).decode()
        # header consists of HDRBLKS x 512 byte blocks
        header_blocks = int(re.findall('\s*HDRBLKS\s*:\s*(\d+)', header_0)[0])
        # read the remaining header
        header = header_0 + f.read(header_blocks * 512 - 512).decode()
        # extract frame info:
        # - rows, cols (NROWS, NCOLS)
        # - bytes-per-pixel of image (NPIXELB)
        # - length of 16 and 32 bit overflow tables (NOVERFL)
        nrows = int(re.findall('\s*NROWS\s*:\s*(\d+)', header)[0])
        ncols = int(re.findall('\s*NCOLS\s*:\s*(\d+)', header)[0])
        npixb = int(re.findall('\s*NPIXELB\s*:\s*(\d+)', header)[0])
        nov16, nov32 = list(map(int, re.findall('\s*NOVERFL\s*:\s*-*\d+\s+(\d+)\s+(\d+)', header)[0]))
        # calculate the size of the image
        im_size = nrows * ncols * npixb
        # bytes-per-pixel to datatype
        bpp2dt = [None, np.uint8, np.uint16, None, np.uint32]
        # set datatype to np.uint32
        data = np.frombuffer(f.read(im_size), bpp2dt[npixb]).astype(np.uint32)
        # read the 16 bit overflow table
        # table is padded to a multiple of 16 bytes
        read_16 = int(np.ceil(nov16 * 2 / 16)) * 16
        # read the table, trim the trailing zeros
        table_16 = np.trim_zeros(np.fromstring(f.read(read_16), np.uint16))
        # read the 32 bit overflow table
        # table is padded to a multiple of 16 bytes
        read_32 = int(np.ceil(nov32 * 4 / 16)) * 16
        # read the table, trim the trailing zeros
        table_32 = np.trim_zeros(np.fromstring(f.read(read_32), np.uint32))
        # assign values from 16 bit overflow table
        data[data == 255] = table_16
        # assign values from 32 bit overflow table
        data[data == 65535] = table_32
        return data.reshape((nrows, ncols))

def read_data(fname, used_only = True):
    '''
    
    '''
    import os
    import numpy as np
    name, ext = os.path.splitext(fname)
    if ext == '.raw':
        data = np.genfromtxt(fname, usecols=(0,1,2,3,4), delimiter=[4,4,4,8,8,4,8,8,8,8,8,8,3,7,7,8,7,7,8,6,5,7,7,7,2,5,9,7,7,4,6,11,3,6,8,8,8,8,4])
    elif ext == '.fco':
        data = np.genfromtxt(fname, skip_header=26, usecols=(0,1,2,4,5,6,7))
        if used_only:
            data = data[data[::,6] == 0]
        data = data[:,[0,1,2,3,4]]
    elif ext == '.sortav':
        data = np.genfromtxt(fname, usecols=(0,1,2,3,6), comments='c')
    elif ext == '.hkl':
        with open(fname) as ofile:
            temp = ofile.readline()
        if len(temp.split()) == 4 and 'NDAT' in temp:
            data = np.genfromtxt(fname, skip_header=1, usecols=(0,1,2,4,5))
        else:
            data = np.genfromtxt(fname, skip_footer=17, usecols=(0,1,2,3,4), delimiter=[4,4,4,8,8,4])
    else:
        data = None
    return data

def get_symmetry_operations(sym = None):
    '''
     
    '''
    import numpy as np
    Symmetry = {  '1':np.array([[[  1,  0,  0],[  0,  1,  0],[  0,  0,  1]]]),
                 '-1':np.array([[[  1,  0,  0],[  0,  1,  0],[  0,  0,  1]],
                                [[ -1,  0,  0],[  0, -1,  0],[  0,  0, -1]]]),
                
                '2/m':np.array([[[  1,  0,  0],[  0,  1,  0],[  0,  0,  1]],
                                [[ -1,  0,  0],[  0,  1,  0],[  0,  0, -1]],
                                [[ -1,  0,  0],[  0, -1,  0],[  0,  0, -1]],
                                [[  1,  0,  0],[  0, -1,  0],[  0,  0,  1]]]),
            
                '222':np.array([[[  1,  0,  0],[  0,  1,  0],[  0,  0,  1]],
                                [[ -1,  0,  0],[  0, -1,  0],[  0,  0,  1]],
                                [[ -1,  0,  0],[  0,  1,  0],[  0,  0, -1]],
                                [[  1,  0,  0],[  0, -1,  0],[  0,  0, -1]]]),
                
                'mmm':np.array([[[  1,  0,  0],[  0,  1,  0],[  0,  0,  1]],
                                [[ -1,  0,  0],[  0, -1,  0],[  0,  0,  1]],
                                [[ -1,  0,  0],[  0,  1,  0],[  0,  0, -1]],
                                [[  1,  0,  0],[  0, -1,  0],[  0,  0, -1]],
                                [[ -1,  0,  0],[  0, -1,  0],[  0,  0, -1]],
                                [[  1,  0,  0],[  0,  1,  0],[  0,  0, -1]],
                                [[  1,  0,  0],[  0, -1,  0],[  0,  0,  1]],
                                [[ -1,  0,  0],[  0,  1,  0],[  0,  0,  1]]]),
            
                '4/m':np.array([[[  1,  0,  0],[  0,  1,  0],[  0,  0,  1]],
                                [[ -1,  0,  0],[  0, -1,  0],[  0,  0,  1]],
                                [[  0,  1,  0],[ -1,  0,  0],[  0,  0,  1]],
                                [[  0, -1,  0],[  1,  0,  0],[  0,  0,  1]],
                                [[ -1,  0,  0],[  0, -1,  0],[  0,  0, -1]],
                                [[  1,  0,  0],[  0,  1,  0],[  0,  0, -1]],
                                [[  0, -1,  0],[  1,  0,  0],[  0,  0, -1]],
                                [[  0,  1,  0],[ -1,  0,  0],[  0,  0, -1]]]),
            
              '4/mmm':np.array([[[  1,  0,  0],[  0,  1,  0],[  0,  0,  1]],
                                [[ -1,  0,  0],[  0, -1,  0],[  0,  0,  1]],
                                [[  0, -1,  0],[  1,  0,  0],[  0,  0,  1]],
                                [[  0,  1,  0],[ -1,  0,  0],[  0,  0,  1]],
                                [[ -1,  0,  0],[  0,  1,  0],[  0,  0, -1]],
                                [[  1,  0,  0],[  0, -1,  0],[  0,  0, -1]],
                                [[  0,  1,  0],[  1,  0,  0],[  0,  0, -1]],
                                [[  0, -1,  0],[ -1,  0,  0],[  0,  0, -1]],
                                [[ -1,  0,  0],[  0, -1,  0],[  0,  0, -1]],
                                [[  1,  0,  0],[  0,  1,  0],[  0,  0, -1]],
                                [[  0,  1,  0],[ -1,  0,  0],[  0,  0, -1]],
                                [[  0, -1,  0],[  1,  0,  0],[  0,  0, -1]],
                                [[  1,  0,  0],[  0, -1,  0],[  0,  0,  1]],
                                [[ -1,  0,  0],[  0,  1,  0],[  0,  0,  1]],
                                [[  0, -1,  0],[ -1,  0,  0],[  0,  0,  1]],
                                [[  0,  1,  0],[  1,  0,  0],[  0,  0,  1]]])}

    if sym in Symmetry.keys():
        return Symmetry[sym]
    else:
        return None

def dict_symmetry_equivalents(data, HKL, symop):
    '''
     TO_CHECK: can loops be merged?
    '''
    import numpy as np
    for r in data:
        h, k, l, Io, Is = r[:5]
        hkl = tuple(np.unique(np.array([h,k,l]).dot(symop), axis=0)[0])
        if hkl in HKL:
            if 'I' in HKL[hkl]:
                HKL[hkl]['I'].append(Io)
                HKL[hkl]['s'].append(Is)
            else:
                HKL[hkl]['I'] = [Io]
                HKL[hkl]['s'] = [Is]
        else:
            HKL[hkl] = {'I':[Io], 's':[Is]}
    return HKL

def calculate_statistics(HKL_1, HKL_2):
    hkl = []
    I1  = []
    I2  = []
    s1  = []
    s2  = []
    for (h,k,l) in HKL_1:
        if (h,k,l) in HKL_2:
            I1.append(np.mean(HKL_1[(h,k,l)]['I']))
            I2.append(np.mean(HKL_2[(h,k,l)]['I']))
            s1.append(np.mean(HKL_1[(h,k,l)]['s']))
            s2.append(np.mean(HKL_2[(h,k,l)]['s']))
            hkl.append((h,k,l))
        else:
            print('> unmatched: ({:3}{:3}{:3}) {}'.format(int(h), int(k), int(l), HKL_1[(h,k,l)]))
    I1  = np.asarray(I1)
    I2  = np.asarray(I2)
    s1  = np.asarray(s1)
    s2  = np.asarray(s2)
    hkl = np.asarray(hkl)
    return I1, I2, s1, s2, hkl

def plot_data(f1, f2, Is1, Is2, hkl = None, _SAVE = True, _SHOW = False, _TITLE = True):

    mpl.rcParams['figure.figsize']   = [7.08661, 4.42913]
    mpl.rcParams['savefig.dpi']      = 600
    mpl.rcParams['font.size']        = 11
    mpl.rcParams['legend.fontsize']  = 11
    mpl.rcParams['figure.titlesize'] = 11
    mpl.rcParams['figure.titlesize'] = 11
    mpl.rcParams['axes.titlesize']   = 11
    mpl.rcParams['axes.labelsize']   = 11
    mpl.rcParams['lines.linewidth']  = 1
    mpl.rcParams['lines.markersize'] = 4
    mpl.rcParams['xtick.labelsize']  = 8
    mpl.rcParams['ytick.labelsize']  = 8
    
    fig = plt.figure()
    grid = plt.GridSpec(10, 16, wspace=0.0, hspace=0.0)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.95, bottom=0.12)
    
    if _ARGS._SCALE == 0.0:
        scale = np.round(np.nansum(f2) / np.nansum(f1), 3)
    else:
        scale = _ARGS._SCALE
    
    if _TITLE:
        #fig.suptitle('Scalefactor: {:6.3f}, cutoff: {}, symmetry: {}\n1: {}\n2: {}'.format(scale, _ARGS._SIGCUT, _ARGS._LAUE, _ARGS._FILE1, _ARGS._FILE2))
        fig.suptitle('{}'.format(_ARGS._PREFIX))
        fig.subplots_adjust(left=0.10, right=0.99, top=0.90, bottom=0.12)
        
    f1     = f1*scale
    Is1    = Is1*scale
    f1cut  = f1[(Is1 > _ARGS._SIGCUT) & (Is2 > _ARGS._SIGCUT)]
    f2cut  = f2[(Is1 > _ARGS._SIGCUT) & (Is2 > _ARGS._SIGCUT)]
    
    print('#1 MEAN: {:14.3f} MEDIAN: {:14.2f}'.format(np.mean(f1), np.median(f1)))
    print('#2 MEAN: {:14.3f} MEDIAN: {:14.2f}'.format(np.mean(f2), np.median(f2)))
    print('SUM(1) : {:14.3f}'.format(np.sum(f1)))
    print('SUM(2) : {:14.3f}'.format(np.sum(f2)))
    print('SCALE  : {:14.3f}'.format(scale))
    print('CUTOFF : {:14.3f} SCALED: {:14.3f}'.format(_ARGS._SIGCUT, _ARGS._SIGCUT*scale))
    
    p00 = fig.add_subplot(grid[ :2,  :7])
    p01 = fig.add_subplot(grid[ :2, 9: ])
    p1x = fig.add_subplot(grid[4:9, 1: ])
    h1y = fig.add_subplot(grid[4:9, 0  ], sharey=p1x)
    h1x = fig.add_subplot(grid[9  , 1: ], sharex=p1x)
    
    p00.scatter(f1cut, f2cut, s=2, color='#37A0CB')
    p00.plot([0, np.nanmax(f1cut)],[0, np.nanmax(f1cut)], 'k-', lw=1.0)
    p00.set_xlabel(r'$I_{{{}}}$'.format(_ARGS._LABEL1))
    p00.set_ylabel(r'$I_{{{}}}$'.format(_ARGS._LABEL2))
    p00.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    p00.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    x = np.log10(f1cut)
    y = np.log10(f2cut)
    p01.scatter(x, y, s=2, color='#37A0CB')
    p01.plot([np.nanmin(x), np.nanmax(x)],[np.nanmin(x), np.nanmax(x)], 'k-', lw=1.0)
    p01.set_xlabel(r'$\log\left(I_{{{}}}\right)$'.format(_ARGS._LABEL1))
    p01.set_ylabel(r'$\log\left(I_{{{}}}\right)$'.format(_ARGS._LABEL2))
    
    facut = (f1cut + f2cut) / 2.
    x = np.log10(facut)
    
    y = (f1cut - f2cut)/(facut)
    
    p1x_sc = p1x.scatter(x, y, s=2, alpha=1.0, picker=True, color='#37A0CB')
    p1x.plot([np.min(x), np.max(x)], [0,0], 'k-', lw=1.0)
    p1x.xaxis.set_visible(False)
    p1x.yaxis.set_visible(False)
    
    h1y.hist(y[(~np.isnan(y)) & (y<2.) & (y>-2.)], 400, color='#003e5c', histtype='stepfilled', orientation='horizontal')
    h1y.xaxis.set_visible(False)
    h1y.invert_xaxis()
    h1y.spines['top'].set_visible(False)
    h1y.spines['bottom'].set_visible(False)
    h1y.set_ylabel(r'$\left(I_{{{0:}}}\ -\ I_{{{1:}}}\right)\ /\ \left<I_{{{{{0:}}},{{{1:}}}}}\right>$'.format(_ARGS._LABEL1, _ARGS._LABEL2))
    
    h1x.hist(x[~np.isnan(x)], 400, color='#003e5c', histtype='stepfilled', orientation='vertical')
    h1x.yaxis.set_visible(False)
    h1x.spines['left'].set_visible(False)
    h1x.spines['right'].set_visible(False)
    h1x.invert_yaxis()
    h1x.set_xlabel(r'$\log\left(\left<I_{{{{{0:}}},{{{1:}}}}}\right>\right)$'.format(_ARGS._LABEL1, _ARGS._LABEL2))

    if _ARGS._MARK:
        if hkl is not None:
            hklcut = hkl[(Is1 > _ARGS._SIGCUT) & (Is2 > _ARGS._SIGCUT)]
        x_1 = []
        y_1 = []
        x_2 = []
        y_2 = []
        for i,r in enumerate(hklcut):
            h,k,l = map(int, r)
            if abs(h) == 2 and abs(k) == 2 and abs(l) == 0:
                x_1.append(x[i])
                y_1.append(y[i])
            if abs(h) == 3 and abs(k) == 4 and abs(l) == 1:
                x_2.append(x[i])
                y_2.append(y[i])
        p1x.plot(x_1, y_1, ls='', marker='o', ms=3, fillstyle='none', mew=1.0, mec='#ee7f00')
        p1x.plot(x_2, y_2, ls='', marker='o', ms=3, fillstyle='none', mew=1.0, mec='#e2007a')
        p1x.annotate('2 2 0', xy=(np.mean(x_1)-0.1, 0.35), size=10, color='#ee7f00', xycoords='data')
        p1x.annotate('3 4 1', xy=(np.mean(x_2)-0.1, 0.35), size=10, color='#e2007a', xycoords='data')
    
    if _SAVE:
        pname = r'{}_{}_vs_{}_c{}_s{}'.format(_ARGS._PREFIX, _ARGS._LABEL1.replace('\\', ''), _ARGS._LABEL2.replace('\\', ''), _ARGS._SIGCUT, scale)
        plt.savefig(pname + '.pdf', transparent=True)
        plt.savefig(pname + '.png', dpi=600, transparent=True)
    
    if _SHOW:
        from collections import defaultdict
        annotations = defaultdict(list)
        background = fig.canvas.copy_from_bbox(p1x.bbox)
        
        if hkl is not None:
            hklcut = hkl[(Is1 > _ARGS._SIGCUT) & (Is2 > _ARGS._SIGCUT)]
        
        def on_pick(event):
            x = event.mouseevent.x
            y = event.mouseevent.y
            ind = event.ind[0]
            h,k,l = map(int, hklcut[(rIsig[:,0] > _ARGS._SIGCUT) & (rIsig[:,1] > _ARGS._SIGCUT)][ind])
            ann_name = '{:3}{:3}{:3}'.format(h,k,l)
            
            if (event.mouseevent.button == 3 and len(annotations) > 0) or ann_name in annotations:
                annotations[ann_name].remove()
                annotations.pop(ann_name)
                fig.canvas.draw_idle()
                return
                
            annotations[ann_name] = plt.annotate(ann_name, xy=(x,y), size=8, xycoords='figure pixels')
            print(ann_name)
            #fig.canvas.draw_idle()
            p1x.draw_artist(annotations[ann_name])
        
        def update_annot(ind, pos, but):
            h,k,l = map(int, hklcut[(rIsig[:,0] > _ARGS._SIGCUT) & (rIsig[:,1] > _ARGS._SIGCUT)][ind])
            ann_name = '{:3}{:3}{:3}'.format(h,k,l)
            
            if ann_name in annotations:
                if but == 3:
                    annotations[ann_name].remove()
                    annotations.pop(ann_name)
                return
            elif but == 1:
                annotations[ann_name] = plt.annotate(ann_name, xy=pos, size=8, xycoords='figure pixels')
                print(ann_name)
            
        def hover(event):
            if event.inaxes == p1x:
                cont, ind = p1x_sc.contains(event)
                if cont:
                    update_annot(ind["ind"][0], (event.x, event.y), event.button)
                    fig.canvas.draw_idle()
                    
        fig.canvas.mpl_connect('motion_notify_event', hover)
        fig.canvas.mpl_connect('pick_event', on_pick)
        plt.show()

if __name__ == '__main__':
    parser = DefaultHelpParser(description = 'Compare two saint .raw integration files.')
    parser.add_argument('-1',  '--file_1', help='Specify the file path (use \'*\' as wildcard)', metavar='PATH', required=False, default='', type=str, dest='_FILE1')
    parser.add_argument('-2',  '--file_2', help='Specify the file path (use \'*\' as wildcard)', metavar='PATH', required=False, default='', type=str, dest='_FILE2')
    parser.add_argument('-x1', '--label_1', help='Label 1', required=False, default='1', type=str, dest='_LABEL1')
    parser.add_argument('-x2', '--label_2', help='Label 2', required=False, default='2', type=str, dest='_LABEL2')
    parser.add_argument('-p',  '--prefix', help='Prefix', required=False, default='', type=str, dest='_PREFIX')
    parser.add_argument('-c',  '--cutoff', help='I cutoff', required=False, default=100, type=float, dest='_SIGCUT')
    parser.add_argument('-s',  '--scale', help='Scale factor (zero for autoscale)', required=False, default=0.0, type=float, dest='_SCALE')
    parser.add_argument('-l',  '--laue', help='Laue symmetry', required=False, default='1', type=str, dest='_LAUE')
    parser.add_argument('-m',  '--mark', help='Mark hkl (edit .py!)', required=False, default=False, action='store_true', dest='_MARK')
    _ARGS = parser.parse_args()
    
    from collections import OrderedDict
    
    symops = get_symmetry_operations(_ARGS._LAUE)
    
    file1, ext1 = os.path.splitext(_ARGS._FILE1)
    file2, ext2 = os.path.splitext(_ARGS._FILE2)
    if ext1 == ext2 == '.sfrm':
        I1 = Is1 = read_sfrm(_ARGS._FILE1)
        I2 = Is2 = read_sfrm(_ARGS._FILE2)
        plot_data(I1, I2, Is1, Is2)
    else:
        dat1 = read_data(_ARGS._FILE1)
        HKL1 = dict_symmetry_equivalents(dat1, OrderedDict(), symops)
        dat2 = read_data(_ARGS._FILE2)
        HKL2 = dict_symmetry_equivalents(dat2, OrderedDict(), symops)
        I1, I2, s1, s2, hkl = calculate_statistics(HKL1, HKL2)
        Is1 = I1/s1
        Is2 = I2/s2
        plot_data(I1, I2, Is1, Is2, hkl)