import json
import argparse

def main(args):
    change_rate = 0
    query_length = 0
    cnt_success = 0
    with open(args.input_path) as json_file:
        json_data = json.load(json_file)
        for feature in json_data:
            if feature['success_indication'] == "Attack success":
                cnt_success += 1
                change_rate += (feature['num_changes']/feature['num_tokens']) * 100
                query_length += feature['query_length']
    change_rate /= len(json_data)
    query_length /= len(json_data)
    print(f"# of success : {cnt_success} / {len(json_data)}")
    print("Average change rate :",change_rate, "%")
    print("Average query length :",query_length)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required = True)
    args = parser.parse_args()
    main(args)