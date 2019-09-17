import json

if __name__ == '__main__':
    business = "/home/someusername/Nextcloud/workspace/uni/bachelor/bachelor_project/data/business.json"

    with open(business, "r") as fread:
        for line in fread:
            node = json.loads(line)
            del_list = []

            if node['attributes'] is None:
                continue

            for jsonElem in node['attributes']:
                attrib = node['attributes'][str(jsonElem)]

                if attrib in ["False", "u'no'", "'no'", "'none'"] or str(jsonElem) == "BusinessParking":
                    del_list.append(jsonElem)
                    continue

                if attrib.startswith("{"):
                    del_list.append(jsonElem)
                    node[str(jsonElem)] = attrib

            for elem in del_list:
                del node['attributes'][str(elem)]

            print(node['attributes'])
