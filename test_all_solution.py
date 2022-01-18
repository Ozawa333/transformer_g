from multiprocessing import Process
import argparse
import json
import os
from tqdm import tqdm
import apps_evaluation.eval.testing_util as test_util
import numpy as np
import signal
import time

def handler(signum, frame):
    raise AssertionError


def print_results(results:list, args):
    print("interview -> competition -> introductory\n")
    pro_cnt = 0
    str_cnt = 0
    ave_total = 0

    for result in results:
        pro_cnt += 1
        temp_res = result
        str_sig = True
        cas_cnt = 0
        ave_cnt = 0

        for i in temp_res:
            cas_cnt += 1
            if(-2 == i):
                str_sig = False
                break
            elif((-1 == i) or (False == i)):
                str_sig = False
                continue
            elif(True == i):
                ave_cnt += 1
                continue
            else:
                print("WORNG!!!!!!!!!!!!")

        if(True == str_sig):
            str_cnt += 1
        if(cas_cnt != 0):
            ave_total += ave_cnt/cas_cnt

        if(pro_cnt==2999):
            print("interview")
            #print("ave_total:", ave_total, "\nstr_cnt:", str_cnt, "\npro_cnt:", pro_cnt)
            print(f"Test Case Average = {(ave_total/pro_cnt)*100}")
            print(f"Strict Accuracy = {(str_cnt/pro_cnt)*100}")
            print("\n")
            ave_total = 0
            str_cnt = 0

        if(pro_cnt==3999):
            print("competition")
            #print("ave_total:", ave_total, "\nstr_cnt:", str_cnt, "\npro_cnt:", pro_cnt)
            print(f"Test Case Average = {(ave_total/(pro_cnt-3000))*100}")
            print(f"Strict Accuracy = {(str_cnt/(pro_cnt-3000))*100}")
            print("\n")
            ave_total = 0
            str_cnt = 0

        if(pro_cnt==4999):
            print("introductory")
            #print("ave_total:", ave_total, "\nstr_cnt:", str_cnt, "\npro_cnt:", pro_cnt)
            print(f"Test Case Average = {(ave_total/(pro_cnt-4000))*100}")
            print(f"Strict Accuracy = {(str_cnt/(pro_cnt-4000))*100}")
            print("\n")
            ave_total = 0
            str_cnt = 0


def eval_and_save_problems(args, i, results_loc, problems, gpt_codes):
    results = {}
    #print(problems)
    for index, problem in enumerate(tqdm(problems)):
        try:
            if args.debug:
                print(f"\n\nproblem path = {problem}")
            output_str = gpt_codes[index]
        except:
            print("CANNOT FIND OUTPUT_STR FOR", problem)
            continue
        prob_path = os.path.join(args.root, problem)

        if not os.path.exists(args.save):
            os.makedirs(args.save)

        res = []
        if args.debug:
            print(f"\nTesting solution {o_idx}")
        curr_res = [-2]

        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(3)
            curr_res = test_util.run_test(prob_path=prob_path, test=output_str, debug=args.debug)
            signal.alarm(0)
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
            #if not np.all(curr_res):
                #print(f"     {curr_res}")
        except AssertionError as e:
            print(f"test framework exception = {repr(e)}{e}\n")
        finally:
            assert isinstance(curr_res, list)
            results[str(i+index)] = curr_res
            #print("----------------",str(i+index))

        if args.debug:
            print(f"\nHow to read results [-2] = compile error, [-1] = runtime error, [False] = failed test case [True] = passed test case")
            print(f"results = {res}")


    with open("./results_ml/"+str("%04d"%i)+".json", "w") as f:
        f.write(json.dumps(results))

    #global all_results
    #all_results = dict(all_results, **results)



def get_data(args):
    with open(args.test_loc, "r") as f:
        problems = sorted(json.load(f))

    print(len(problems))
    gpt_codes = {}
    gpt_bleu = {}
    gpt_codebleu = {}
    results = {}
    codes_loc = os.path.join(args.save, f"codes.json")
    if not os.path.exists(codes_loc):
        codes_loc = os.path.join(args.save, f"{args.start}-{args.end}_codes-Copy1.json")

    if os.path.exists(codes_loc):
        results_loc = os.path.join(args.save, f"all_results.json") 
    else:
        results_loc = os.path.join(args.save, f"{args.start}-{args.end}_results.json") 
    print(codes_loc, results_loc)

    with open(codes_loc, "r") as f:
        gpt_codes = json.load(f)

    if args.index:
        problems = [problems[args.index]]
    else:
        if args.start > len(problems) or args.start < 0:
            print(f"start index {args.start} > number of problems {len(problems)}")
            return
        start = args.start
        if args.end is None or args.end > len(problems):
            end = len(problems)
        else:
            end = args.end
        problems = problems[start:end]

    if args.stop_early:
        problems = problems[:args.stop_early]

    return problems, gpt_codes, results_loc


parser = argparse.ArgumentParser(description="Testing a Language Model on Python Code")
parser.add_argument("-t","--test_loc", default="./test.json", type=str, help="path to the json containing problem paths to be evaluated.")
parser.add_argument("-r","--root", default="./", type=str, help="where the data is stored.")
parser.add_argument("-s","--start", default=0, type=int)
parser.add_argument("-e","--end", default=None, type=int, help="If you want to evaluate a subset of problems specify start and ending index. File with start and ending prefix must exist typically used with batch evaluation.")
parser.add_argument("-i", "--index", default=0, type=int)
parser.add_argument("-d", "--debug", default=False, action="store_true")
parser.add_argument("--save", type=str, default="./", help="Where the evaluated data is loaded from and results saved to.")
parser.add_argument("--stop-early", default=None, type=int)

parser.add_argument("-g", "--get_results", default=False)
parser.add_argument("-p", "--print_results", action="store_true", help="If you have already evaluated the results and only want to print them.")

args = parser.parse_args()

if(args.get_results):
    a = []
    codes = []
    problems, gpt_codes, results_loc = get_data(args)

    for value in gpt_codes.values():
        codes.append(value)
    #print(codes[:10])

    for i in range(10):
        for j in range(10):
            for k in range(10):
                a.append(Process(target=eval_and_save_problems,
                                 args=(args, i*500+j*50+k*5, results_loc, 
                                       problems[(i*500+j*50+k*5):(i*500+j*50+(k+1)*5)], 
                                       codes[(i*500+j*50+k*5):(i*500+j*50+(k+1)*5)], )))
                a[i*100+j*10+k].start()
                time.sleep(k*0.1)
                #break
            time.sleep(j)
            #break
        time.sleep(i*10)
        #break


if(args.print_results):
    filePath = './results_ml/'

    dirlist= os.listdir(filePath)
    #print(dirlist)

    results = []
    cont = 0
    for file in dirlist:
        if(file[-5:] == ".json"):
            with open(filePath+file, "r") as f: 
                temp = json.load(f)
                for value in temp.values():
                    #print(value)
                    cont += 1
                    results.append(value)

    print("total questions:" ,cont)
    #print(type(results))
    #for index in results

    print_results(results, args)


'''
with open("./results_ml/all"+".json", "w") as f:
    f.write(json.dumps(all_results))
print(1)
'''

