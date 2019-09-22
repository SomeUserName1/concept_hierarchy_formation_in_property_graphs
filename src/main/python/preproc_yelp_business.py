import json
import ast


def prune(node, key):
    val_list = []
    value = node[str(key)]

    if str(value).startswith("{"):
        dictionary = ast.literal_eval(value)
        for subkey, value in dictionary.items():
            if value not in false_values:
                val_list.append(subkey)
        node[str(key)] = val_list

    if node[str(key)] is None or len(node[str(key)]) == 0 or value in ["None", "False", False]:
        del node[str(key)]


def main():
    with open(business, "r") as fread, open(preproc_business, "w") as fwrite:
        extra_fields = set()
        for line in fread:
            node = json.loads(line)
            attrib_list = []

            if node['attributes'] is None:
                continue

            for key, value in node['attributes'].items():

                if value in false_values:
                    continue
                elif value in true_values:
                    attrib_list.append(key)
                else:
                    node[str(key)] = value
                    prune(node, key)
                    extra_fields.add(key)
            node['attributes'] = attrib_list

            if len(node['attributes']) == 0:
                del node['attributes']

            del node['is_open']
            del node['hours']

            node['categories'] = list(node['categories'].split(","))

            fwrite.write(json.dumps(node) + "\n")


if __name__ == '__main__':
    business = "/home/someusername/Nextcloud/workspace/uni/bachelor/bachelor_project/data/business.json"
    preproc_business = "/home/someusername/Nextcloud/workspace/uni/bachelor/bachelor_project/data/business_preproc.json"
    false_values = ["False", "u'no'", "'no'", "'none'", "None", None, False, "u'none'", "u'outdoor'", "'outdoor'"]
    true_values = ["True", "u'yes'", True, "yes", "'yes_corkage'", "'yes_free'", "'yes'", "u'yes_free'", "u'yes_corkage'"]
    main()