#script to convert byte arrays into images. images are saved in the format [nightid]_[candid]_[sci/ref/diff]_image.jpg

import io
import gzip
import matplotlib.pyplot as plt
from astropy.io import fits
import aplpy

nightid= 268; #for a particular night

#plot_cutout function for plotting the byte array
def plot_cutout(stamp, fig=None, subplot=None, **kwargs):
    with gzip.open(io.BytesIO(stamp), 'rb') as f:
        with fits.open(io.BytesIO(f.read())) as hdul:
            if fig is None:
                fig = plt.figure(figsize=(4,4))
            if subplot is None:
                subplot = (1,1,1)
            ffig = aplpy.FITSFigure(hdul[0],figure=fig, subplot=subplot, **kwargs)
            ffig.show_grayscale(stretch='arcsinh')
    return ffig


#getting byte array from database
import psycopg2
connection = psycopg2.connect("dbname=gattinibot user=gdbadmin password=4gattiniDBs! host=gattinidrp.caltech.edu")
connection
import psycopg2.extras
cursor = connection.cursor(cursor_factory = psycopg2.extras.DictCursor)
cursor
query = 'select * from cutouts cut, candidates cand where cut.candid = cand.candid and nightid= 268;'
cursor.execute(query)
out = cursor.fetchall()


count= 0 #counter

for i in out:
        candid= out[count]['candid']

        #calling function to plot science image
        filename= str(nightid)+'_'+str(candid)+'_'+'sci_image'
        plot_cutout(out[count]['sci_image']).savefig('jpeg_images/'+filename+'.jpg')

        #reference image
        filename= str(nightid)+'_'+str(candid)+'_'+'ref_image'
        plot_cutout(out[count]['ref_image']).savefig('jpeg_images/'+filename+'.jpg')

        #difference image
        filename= str(nightid)+'_'+str(candid)+'_'+'diff_image'
        plot_cutout(out[count]['diff_image']).savefig('jpeg_images/'+filename+'.jpg')

        count= count+1

print(count)
