import sys

def main(argv):
	col = int(argv[0])
	fin = open(argv[1], 'r')
	fout = open('ans1.txt', 'w')
	ct = 0
	numarr = []
	for line in fin.readlines():
		numarr.append([])
		arr = line.strip(' \n').split(' ')
		for i in range(len(arr)):
			numarr[ct].append(float(arr[i]))
		ct += 1
	ans = [numarr[i][col] for i in range(ct)]
	ans.sort()
	for i in range(len(ans)):
		outstr = '{0}'.format(ans[i]) if i==0 else ',{0}'.format(ans[i])
		fout.write(outstr)

if __name__ == '__main__':
	main(sys.argv[1:])
