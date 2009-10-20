import re
import heapq
import operator
import math
import statlib.stats

comment_pattern = re.compile("//.*")

def tobin(x, count=8):
        """
        Integer to binary
        Count is number of bits
        """
        return "".join(map(lambda y:str((x>>y) & 1), range(count-1, -1, -1)))

def parse(problem_set):
        data = {}
        columns = {}
        with open("../prog1data/{0}/{0}.names".format(problem_set)) as names_file:
                i = 0
                for line in names_file:
                        line = line.strip()
#            match = comment_pattern.search(line)
                        line = re.sub(comment_pattern, "", line)
                        if line:
                                line = re.sub(r"\s+", "", line)
                                line = line.strip(".")
                                if i == 0:
                                        #First line has to be class labels
                                        if "0,1" in line:
                                                i += 1
                                                continue
                                else:
                                        label, values = line.split(":")
                                        columns[i-1] = (label, values.split(","))
                                        i += 1
                columns[len(columns)] = ("class label", ["0","1"])

        with open("../prog1data/{0}/{0}.data".format(problem_set)) as data_file:
                for line in data_file:
                        line = line.strip()
                        line = re.sub(comment_pattern, "", line)
                        if line:
                                line = re.sub(r"\s+", "", line)
                                line = line.strip(".")
                                parts = line.split(",")
                                if parts[0] in columns[0][1]:
                                        data[parts[0]] = []
                                else:
                                        continue
                                for i in range(1,len(parts)):
                                        part = parts[i]
                                        if part in columns[i][1]:
                                                data[parts[0]].append(part)
                                        elif "continuous" in columns[i][1]:
                                                try:
                                                        part = float(part)
                                                        data[parts[0]].append(part)
                                                except ValueError:
                                                        if part in "?":
                                                                data[parts[0]].append(None)
                                        elif part in "?":
                                                data[parts[0]].append(None)
        return (columns, data)

def parse_to_logn_and_normalize(problem_set):
        columns, data = parse(problem_set)
        for key in columns:
                info = columns[key]
                temp = list(info[:])
                temp.append(len(info[1]))
                columns[key] = tuple(temp)

        #Calculate the attribute statistics so we can figure out the most common attribute
        stats = {}
        for key in data:
                example = data[key]
                for i in range(0, len(example)-1):
                        if not stats.get(i):
                                stats[i] = {}
                        if example[i]:
                                if stats[i].get(example[i]):
                                        stats[i][example[i]] += 1
                                else:
                                        stats[i][example[i]] = 1

        #Calculate the value that should be filled in for any missing values
        missing_values= {}
        for i in range(0, len(example)-1):
                possibles = stats[i]
                if "continuous" in columns[i+1][1]:
                        temp = [x * y for x, y in possibles.iteritems()]
                        temp2 = sum(possibles.itervalues())
                        temp = math.fsum(temp)
                        missing_values[i] = temp / temp2
                else:
                        most_common = heapq.nlargest(1, possibles.iteritems(), operator.itemgetter(1))
                        missing_values[i] = most_common[0][0]

        #Fill in any missing values
        for key in data:
                example = data[key]
                for i in range(0, len(example)-1):
                        if not example[i]:
                                example[i] = missing_values[i]

        normalizers = {}
        #Calc Normalization Params
        for i in range(0, len(example)-1):
                if "continuous" in columns[i+1][1]:
                        temp = [example[i] for example in data.itervalues()]
                        normalizers[i] = (statlib.stats.lmean(temp), statlib.stats.lstdev(temp))

        #Create Log(N) mappings
        mapping = {}
        for i in range(0, len(example)-1):
                if "continuous" not in columns[i+1][1]:
                        values = columns[i+1][1]
                        num_values = columns[i+1][2]
                        num_inputs = math.ceil(math.log(num_values, 2))
                        for i in range(0, len(values)):
                                mapping[values[i]] = [int(x) for x in tobin(i, num_inputs)]
        
        #Convert to log(N) encoding and normalize
        for key in data:
                example = data[key]
                new_example = []
                for i in range(0, len(example)-1):
                        if "continuous" in columns[i+1][1]:
                                val = example[i]
                                new_val = (val - normalizers[i][0]) / normalizers[i][1]
                                new_example.append(new_val)
                        else:
                                new_example.extend(mapping[example[i]])
                new_example.append(int(example[-1]))
                data[key] = new_example
        
        return(columns, data)

