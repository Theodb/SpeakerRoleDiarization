from sklearn.preprocessing import LabelEncoder

class LabelRTTM():
    def __init__(self, fileName=None, startTime=None, duration=None, speakerName=None, rttmLine=None):
        if rttmLine:
            self.load(rttmLine)
        else:
            self.id = fileName.split(".")[0]
            self.startTime = float(startTime)
            self.duration = float(duration)
            self.endTime = self.startTime + self.duration
            self.speakerName = speakerName

    def load(self, rttmLine):
        line = rttmLine.split()
        self.id = line[1].split(".")[0]
        self.startTime = float(line[3])
        self.duration = float(line[4])
        self.endTime = self.startTime + self.duration
        self.speakerName = line[7]

    def format_rttm(self):
        return 'SPEAKER {0} 1 {1} {2} <NA> <NA> {3} <NA> <NA>\n'.format(self.id, self.startTime, self.duration, self.speakerName)


class ProcessRTTM():
    def __init__(self, path, load=False, encode=False, elimOverlap=False):
        self.path = path
        self.lines = []
        self.rttmLines = []
        self.speakerCount = 0
        if load:
            self.loadFile()
        if encode:
            self.loadFile()
            self.encode_rttm()
            self.countSpeaker()
        if elimOverlap:
            self.loadFile()
            self.encode_rttm()
            self.countSpeaker()
            self.eliminateOverlap()

    def loadFile(self):
        with open(self.path) as file:
            self.lines = [line.rstrip() for line in file.readlines()]
    def getStartTime(self):
        return self.rttmLines[0].startTime

    def getEndTime(self):
        return sorted(self.rttmLines, key=lambda x: x.endTime, reverse=True)[0].endTime

    def encode_rttm(self):
        self.rttmLines = [LabelRTTM(rttmLine=line) for line in self.lines]

    def countSpeaker(self):
        self.speakerCount = len(
            set([speaker.speakerName for speaker in self.rttmLines]))

    def eliminateOverlap(self):
        """
        # for 2 speaker only
        newLines = []
        start = lines[0].startTime
        for i in range(0,len(lines)):
            if i == len(lines)-1:
                if start < lines[i].endTime:
                    newLines.append(LabelRTTM(startTime=start, duration=lines[i].endTime - start, speakerName=lines[i].speakerName, fileName=lines[i].id ))
            end = min([k.startTime for k in lines[i+1:i+speakerCount]])
            if lines[i].endTime > lines[i+1].startTime:
                if start < end:
                    newLines.append(LabelRTTM(startTime=start, duration=end - start, speakerName=lines[i].speakerName, fileName=lines[i].id ))
                start = lines[i].endTime
            else:
                newLines.append(lines[i])
                start = lines[i+1].startTime
        """
        newLines = []
        start = self.rttmLines[0].startTime
        for i in range(0, len(self.rttmLines)):
            if i == len(self.rttmLines)-1:
                if start < self.rttmLines[i].endTime:
                    newLines.append(LabelRTTM(
                        startTime=start, duration=self.rttmLines[i].endTime - start, speakerName=self.rttmLines[i].speakerName, fileName=self.rttmLines[i].id))
                break
            end = min(
                self.rttmLines[i+1:i+self.speakerCount], key=lambda k: k.startTime)
            if self.rttmLines[i].endTime > end.startTime:
                if start < end.startTime:
                    newLines.append(LabelRTTM(startTime=start, duration=end.startTime - start,
                                    speakerName=self.rttmLines[i].speakerName, fileName=self.rttmLines[i].id))
                start = self.rttmLines[i].endTime
            else:
                newLines.append(self.rttmLines[i])
                start = end.startTime
        self.rttmLines = newLines

    def save(self, path):
        f = open(path, 'w')
        f.write(''.join([i.format_rttm() for i in self.rttmLines]))
        f.close()

    def getSpeakers(self):
        return list(set(map(lambda x:x.speakerName, self.rttmLines)))
		
    def filterByDuration(self, duration):
        return [i for i in self.rttmLines if i.duration > duration]
    
    def filterByinterval(self, start, end):
        newRttmLines = []
        rttmLines = self.rttmLines.copy()
        for i in range(len(rttmLines)):
            if rttmLines[i].endTime > start and rttmLines[i].startTime < end:
                if rttmLines[i].startTime < start:
                    rttmLines[i].startTime = start
                if rttmLines[i].endTime > end:
                    rttmLines[i].endTime = end
                newRttmLines.append(rttmLines[i])
        return newRttmLines
        
    def labelEncode(self):
        return LabelEncoder().fit_transform([line.speakerName for line in self.rttmLines])