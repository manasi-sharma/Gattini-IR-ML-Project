#!/scr2/kde/packages/anaconda3/bin/python

#script to create a manifest of all of the candidates from a particular night and associated metadata for uploading to Zooniverse

import io
import csv

nightid= 268; #for a particular night


#getting byte array from database
import psycopg2
connection = psycopg2.connect("dbname=gattinibot user=gdbadmin password=4gattiniDBs! host=gattinidrp.caltech.edu")
connection
import psycopg2.extras
cursor = connection.cursor(cursor_factory = psycopg2.extras.DictCursor)
cursor
query = 'select * from cutouts cut, candidates cand where cut.candid = cand.candid and nightid= '+str(nightid)+';'
cursor.execute(query)
out = cursor.fetchall()
print('number of observations: '+str(len(out)))

count= 0 #counter

#to clear csv file
f = open("manifest.csv", "w+")
f.close()

#to write header for the csv file
with open('manifest.csv', mode='a') as manifest:
    manifest_writer = csv.writer(manifest, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    manifest_writer.writerow(['counter', 'candid', 'sci_image', 'ref_image', 'diff_image', 'nightid', 'scorr_peak'])

#to write all the entries
for i in out:
        print(count)

        candid= out[count]['candid']
        scorr_peak= out[count]['scorr_peak']
        filename= str(nightid)+'_'+str(candid)+'_'

        with open('manifest.csv', mode='a') as manifest:
                manifest_writer = csv.writer(manifest, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                manifest_writer.writerow([str(count), str(candid), filename+'sci_image', filename+'ref_image', filename+'diff_image', str(nightid), str(scorr_peak)])

        count= count+1

manifest.close()
