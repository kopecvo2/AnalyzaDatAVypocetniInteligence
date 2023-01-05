##
import csv
import numpy as np
from scipy.fft import fft, fftfreq


def normalize(data):
    data = np.abs(data)
    data = np.log(data)
    data = (data-np.mean(data))/np.std(data)
    return data

f = 1024
bias = 0

# unbroken_file = open('C://Users//vojta//Documents//GitHub//MKP_Diplomka//unbroken_pokus.csv')
# broken_file = open('C://Users//vojta//Documents//GitHub//MKP_Diplomka//broken_pokus.csv')
# reference_file = open("C://Users//vojta//Documents//GitHub//MKP_Diplomka//reference_pokus.csv")

# unbroken_file = open('C://Users//vojta//Documents//GitHub//MKP_Diplomka//verification_unbroken.csv')
# broken_file = open('C://Users//vojta//Documents//GitHub//MKP_Diplomka//verification_broken.csv')
# reference_file = open("C://Users//vojta//Documents//GitHub//MKP_Diplomka//verification_reference.csv")

unbroken_file = open('C:/Users/vojta/Documents/GitHub/MKP_Diplomka/unbroken_Lko_2.csv')
broken_file = open('C:/Users/vojta/Documents/GitHub/MKP_Diplomka/broken_Lko_2.csv')
reference_file = open("C:/Users/vojta/Documents/GitHub/MKP_Diplomka/reference_Lko_2.csv")

refread = list(csv.reader(reference_file))
brokread = list(csv.reader(broken_file))
unbrokread = list(csv.reader(unbroken_file))
#
for i in np.arange(0, len(refread)//2):
    refrow = np.array(refread[i*2])
    brokrow = np.array(brokread[i * 2])
    unbrokrow = np.array(unbrokread[i * 2])
    ref_fft = fft(refrow)
    brok_fft = fft(brokrow)
    unbrok_fft = fft(unbrokrow)
    freq = fftfreq(ref_fft.size, d=(1 / f))
    freq = freq[0:f]
    ref_fft = ref_fft[0:f]
    brok_fft = brok_fft[0:f]
    unbrok_fft = unbrok_fft[0:f]

    file = open(f'input_data_Lko/reference/reference_{bias+i}.csv', 'w')
    writer = csv.writer(file)
    writer.writerow(normalize(ref_fft))
    file.close()

    file = open(f'input_data_Lko/broken/broken_{bias+i}.csv', 'w')
    writer = csv.writer(file)
    writer.writerow(normalize(brok_fft))
    file.close()

    file = open(f'input_data_Lko/unbroken/unbroken_{bias+i}.csv', 'w')
    writer = csv.writer(file)
    writer.writerow(normalize(unbrok_fft))
    file.close()

    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    # ax1.plot(freq, np.abs(ref_fft))
    # ax1.grid()
    # ax2.plot(freq, normalize(unbrok_fft))
    # ax2.grid()
    # ax3.plot(freq, np.abs(brok_fft))
    # ax3.grid()
    # plt.show()

print('Done')
print(f'Set new bias to: {bias+i+1}')
