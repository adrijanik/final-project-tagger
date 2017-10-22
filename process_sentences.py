import os
def process_sentences():    
    path = "./test_sentences/"
    files = list(filter(lambda x: "dot" in x, os.listdir(path)))
    for fl in files:
        lines = []
        with open(path + fl, 'r') as f:
            for line in f:
                if 'weight' in line:
                    split = line.split(' ')
                    if len(split) == 5:
                        lines.append('"'+split[0]+'" '+split[1]+' "'+split[2]+'" '+split[4])
                    else:
                        lines.append('"'+split[0]+'" '+split[1]+' "'+split[2]+'" '+split[3])
    
        with open(path+'new_'+fl, 'w') as fw:
            fw.write('digraph "" {\n'+"".join(lines)+ '}')
    
