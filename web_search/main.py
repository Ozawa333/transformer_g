import json
import requests

# Currently use ENCODER_SEARCH_ENGINE_ID only

API_KEY = "AIzaSyDpnhTpbzPE_kw4-34E_ZNBwrff-pJNgo4"
ENCODER_SEARCH_ENGINE_ID = "bd40f442cc1318d85"
DECODER_EARCH_ENGINE_ID = "8e1edf37eecc27858"
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


def make_dict(name):
    with open("./"+name+"_description.json", "r") as load_f:
        tokens = json.load(load_f)
    search_token = ""
    # print(tokens)
    for i,token in enumerate(tokens):
        # print(token)

        if(token.isdigit()):
            tokens[token] = token

        if(isinstance(tokens[token], int)):
            if("Ġ" == token[0]):
                search_token = token[1:]
            print("----------"+str(i)+"----------")
            print(token)
            tokens[token] = search(API_KEY, ENCODER_SEARCH_ENGINE_ID, search_token, start)
            print(tokens[token])

        if(i%10 == 0):
            #print(type(tokens))
            tokens_json = json.dumps(tokens)
            with open("./"+name+"_description.json", "w") as save_f:
                save_f.write(tokens_json)


'''make'''
# make_json("inputs")
# make_json("outputs")

'''search'''
make_dict("inputs")
# make_dict("outputs")


