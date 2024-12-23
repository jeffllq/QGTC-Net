import datetime

logmsg = ''
timemark = dict()
saveDefault = False
def log(msg, save=None, oneline=False):
	global logmsg
	global saveDefault
	time = datetime.datetime.now()
	formatted_time = time.strftime('%Y-%m-%d %H:%M:%S')
	tem = '%s: %s' % (formatted_time, msg)
	if save != None:
		if save:
			logmsg += tem + '\n'
	elif saveDefault:
		logmsg += tem + '\n'
	if oneline:
		print(tem, end='\r')
	else:
		print(tem)

def marktime(marker):
	global timemark
	timemark[marker] = datetime.datetime.now()


if __name__ == '__main__':
	log('')