import json
import requests

# https://console.cloud.google.com/home/dashboard
# https://developers.google.com/custom-search/v1/overview
API_KEY_S = [
            ]

SEARCH_ENGINE_ID = "bd40f442cc1318d85"
# DECODER_EARCH_ENGINE_ID = "8e1edf37eecc27858"
query = "1"
page = 1
start = (page - 1) * 10 + 1


def make_json(name):
    with open("../model/"+name+"_tokenizer.json", "r") as load_f:
        load_dict = json.load(load_f)
        tokens = dict(load_dict["model"]["vocab"])
        print(tokens)
        tokens = json.dumps(tokens)
    with open("./"+name+"_description.json", "w") as save_f:
        save_f.write(tokens)


def search(API_KEY, SEARCH_ENGINE_ID, query, start):
    #print(API_KEY,SEARCH_ENGINE_ID, query, start)
    url = f"https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={SEARCH_ENGINE_ID}&q={query}&start={start}"

    # make the API request
    data = requests.get(url).json()
    search_items = data.get("items")
    snippets = []
    # print(search_items)

    if(search_items != None):
        for i, search_item in enumerate(search_items, start=1):
            snippet = search_item.get("snippet")
            snippets.append(snippet)
        return snippets
    else:
        return None


def make_dict(name, API_KEY, SEARCH_ENGINE_ID):
    with open("./"+name+"_description.json", "r") as load_f:
        tokens = json.load(load_f)
    search_token = ""
    # print(tokens)

    search_count = 0
    for i,token in enumerate(tokens):
        searched_singnal = False
        print("----------"+str(i)+"----------")
        print(token)

        if(token.isdigit()):
            tokens[token] = token
            print("digit")
            continue

        if(isinstance(tokens[token], int)):
            for searched_token, _ in tokens.items():
                if((searched_token==token[1:] or searched_token[1:]==token) and ("Ġ" == searched_token[0] or "Ġ" == token[0]) and len(token)>1):
                    tokens[token] = tokens[searched_token]
                    print("searched")
                    searched_singnal = True
                    break

            if(searched_singnal == True):
                break

            if("Ġ" == token[0]):
                search_token = token[1:]
            else:
                search_token = token

            tokens[token] = search(API_KEY, SEARCH_ENGINE_ID, search_token, start)
            search_count += 1
            print(tokens[token])

        if(search_count > 99):
            break

    #print(type(tokens))
    tokens_json = json.dumps(tokens)
    with open("./"+name+"_description.json", "w") as save_f:
        save_f.write(tokens_json)


'''make'''
# make_json("inputs")
# make_json("outputs")

'''search'''
for API_KEY in API_KEY_S:
    #make_dict("inputs", API_KEY, SEARCH_ENGINE_ID)
    make_dict("outputs", API_KEY, SEARCH_ENGINE_ID)


