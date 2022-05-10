
import sys
import numpy
import matplotlib
import matplotlib.pyplot as plt

rg = [0, 0]
with open("sample.range") as f:
    for line in f.readlines():
        line_arr = line.split(" ")
        rg[0] = int(line_arr[1], 16)
        rg[1] = int(line_arr[3], 16)
        break

addr_max = 0
addr_min = sys.maxsize
addr2_max = 0
addr2_min = sys.maxsize

addrs = []
addrs2 = []
with open("sample.log") as f:
    for line in f.readlines():
        line_arr = line.split(" ")
#        addra = int(line_arr[6], 16)
#        addrb = int(line_arr[10], 16)
#        if addrb >= rg[0] and addrb < rg[1]:
#            addrs.append(addrb)
        # if addrb % 4096 == 137:
#            print("{}:{}".format(int(addrb / 4096), addrb % 4096))

        int_str = line_arr[2].split(",")[0]
        int_val = int(int_str, 16)
        if int_val <= (1 << 28):
            addrs.append(int_val)
        else:
            addrs2.append(int_val)

for addr in addrs:
    if addr > addr_max:
        addr_max = addr
    if addr < addr_min:
        addr_min = addr

for addr in addrs2:
    if addr > addr2_max:
        addr2_max = addr
    if addr < addr2_min:
        addr2_min = addr

page_max = int(addr_max / 4096)
page_min = int(addr_min / 4096)
page_max2 = int(addr2_max / 4096)
page_min2 = int(addr2_min / 4096)

print("page_range:",page_min,page_max)
print("page_range2:", page_min2, page_max2)

num_page = page_max - page_min + 1
num_page2 = page_max2 - page_min2 + 1
arr = numpy.zeros((num_page, int(4096 / 64)))
arr2 = numpy.zeros((num_page2, int(4096 / 64)))

for addr in addrs:
    page_num = int(addr / 4096)
    page_off = int((addr % 4096) / 64)
    arr[page_num - page_min, page_off] = 1
    for i in range(10):
         if page_num - page_min + i < num_page:
             arr[page_num - page_min + i, page_off] = 1
#    if page_num - page_min - 1 >= 0:
#        arr[page_num - page_min - 1, page_off] += 10
#    if page_num - page_min + 1 < page_max:
#        arr[page_num - page_max + 1, page_off] += 10
    # print(page_num - page_min, page_off)

#arr = numpy.minimum(arr, 1)

for addr in addrs2:
    page_num = int(addr / 4096)
    page_off = int((addr % 4096) / 64)
    arr2[page_num - page_min2, page_off] = 1
    for i in range(10):
         if page_num - page_min2 + i < num_page2:
             arr2[page_num - page_min2 + i, page_off] = 1

print(arr.shape,arr2.shape)
print(len(addrs),len(addrs2))

#print(arr[:,2])
fig, ax = plt.subplots()
fig.set_size_inches(6.4 * 12, 4.8 * 6)
im = ax.imshow(numpy.transpose(numpy.concatenate((arr,arr2))),cmap='gray',interpolation='none')
ax.set_aspect(100)
fig.tight_layout()
plt.savefig("sample.pdf", format="pdf")
