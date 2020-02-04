import numpy as np
import glob, argparse, os
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description = 'Compare two saint .raw integration files.')
parser.add_argument('-1',  '--file_1', help='Specify the file path (use \'*\' as wildcard)', metavar='PATH', required=False, default='', type=str, dest='_FILE1')
parser.add_argument('-2',  '--file_2', help='Specify the file path (use \'*\' as wildcard)', metavar='PATH', required=False, default='', type=str, dest='_FILE2')
parser.add_argument('-l1', '--label_1', help='Label 1', required=False, default='1', type=str, dest='_LABEL1')
parser.add_argument('-l2', '--label_2', help='Label 2', required=False, default='2', type=str, dest='_LABEL2')
parser.add_argument('-p',  '--prefix', help='Prefix', required=False, default='', type=str, dest='_PREFIX')
parser.add_argument('-c',  '--cutoff', help='I cutoff', required=False, default=100, type=float, dest='_CUTOFF')
parser.add_argument('-s',  '--scale', help='Calculate scale factor', required=False, default=False, action='store_true', dest='_SCALE')
_ARGS = parser.parse_args()

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

f1 = read_sfrm(_ARGS._FILE1)
f2 = read_sfrm(_ARGS._FILE2)
c1 = f1[f2 > _ARGS._CUTOFF]
c2 = f2[f2 > _ARGS._CUTOFF]

scale = 1.0
if _ARGS._SCALE:
    #scale = np.sum(np.multiply(s1, s2)) / np.sum(np.multiply(s1, s1))
    scale = np.round(np.sum(c2) / np.sum(c1), 3)

cutoff = _ARGS._CUTOFF / scale

print('#1 MEAN: {:14.3f} MEDIAN: {:10.2f}'.format(np.mean(f1), np.median(f1)))
print('#2 MEAN: {:14.3f} MEDIAN: {:10.2f}'.format(np.mean(f2), np.median(f2)))
print('SUM(s1): {:14.3f}'.format(np.sum(f1)))
print('SUM(s2): {:14.3f}'.format(np.sum(f2)))
print('SCALE  : {:14.3f}'.format(scale))
print('CUTOFF : {:14.3f} SCALED: {:11.3f}'.format(_ARGS._CUTOFF, cutoff))

_PREFIX  = _ARGS._PREFIX
_LABEL_1 = _ARGS._LABEL1
_LABEL_2 = _ARGS._LABEL2

d1 = f1
d2 = f2 / scale
c1 = d1[(d2 > cutoff) & (d1 > cutoff)]
c2 = d2[(d2 > cutoff) & (d1 > cutoff)]

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

p00 = fig.add_subplot(grid[ :2,  :7])
p01 = fig.add_subplot(grid[ :2, 9: ])
p1x = fig.add_subplot(grid[4:9, 1: ])
h1y = fig.add_subplot(grid[4:9, 0  ], sharey=p1x)
h1x = fig.add_subplot(grid[9  , 1: ], sharex=p1x)

p00.scatter(c1, c2, s=2, color='#37A0CB')
p00.plot([0, np.nanmax(c1)],[0, np.nanmax(c1)], 'k-', lw=1.0)
p00.set_xlabel(r'$I_{{{}}}$'.format(_LABEL_1))
p00.set_ylabel(r'$I_{{{}}}$'.format(_LABEL_2))
p00.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
p00.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

x = np.log10(c1)
y = np.log10(c2)
p01.scatter(x, y, s=2, color='#37A0CB')
p01.plot([np.nanmin(x), np.nanmax(x)],[np.nanmin(x), np.nanmax(x)], 'k-', lw=1.0)
p01.set_xlabel(r'$\log\left(I_{{{}}}\right)$'.format(_LABEL_1))
p01.set_ylabel(r'$\log\left(I_{{{}}}\right)$'.format(_LABEL_2))

x = np.log10((c1 + c2) / 2.0)
y = (c1 - c2) / ((c1 + c2) / 2.0)
p1x_sc = p1x.scatter(x, y, s=2, alpha=1.0, picker=True, color='#37A0CB')
p1x.plot([np.min(x), np.max(x)], [0,0], 'k-', lw=1.0)
p1x.xaxis.set_visible(False)
p1x.yaxis.set_visible(False)

h1x.hist(x[~np.isnan(x)], 400, color='#003e5c', histtype='stepfilled', orientation='vertical')
h1x.yaxis.set_visible(False)
h1x.spines['left'].set_visible(False)
h1x.spines['right'].set_visible(False)
h1x.invert_yaxis()
h1x.set_xlabel(r'$\log\left(\left<I_{{{{{0:}}},{{{1:}}}}}\right>\right)$'.format(_LABEL_1, _LABEL_2))

h1y.hist(y[(~np.isnan(y)) & (y<2.) & (y>-2.)], 400, color='#003e5c', histtype='stepfilled', orientation='horizontal')
h1y.xaxis.set_visible(False)
h1y.invert_xaxis()
h1y.spines['top'].set_visible(False)
h1y.spines['bottom'].set_visible(False)
h1y.set_ylabel(r'$\left(I_{{{0:}}}\ -\ I_{{{1:}}}\right)\ /\ \left<I_{{{{{0:}}},{{{1:}}}}}\right>$'.format(_LABEL_1, _LABEL_2))

pname = '{}_{}_vs_{}_c{}_s{}'.format(_PREFIX, _LABEL_1.replace('\\', ''), _LABEL_2.replace('\\', ''), _ARGS._CUTOFF, scale)
#plt.savefig(pname + '.pdf', transparent=True)
plt.savefig(pname + '.png', dpi=600, transparent=True)