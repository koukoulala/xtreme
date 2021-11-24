import sys

def deal_data(ads_path, changed_ads_path):
    fw = open(changed_ads_path, "w", encoding='utf-8')

    with open(ads_path, "r", encoding='utf-8') as fr:
        for line in fr.readlines():
            line = line.strip('\n').split('\t')
            text_a = line[2]
            text_b = line[3]
            label = line[1]

            fw.write(text_a + "\t" + text_b + "\t" + label + "\n")

    fw.close()


if __name__ == '__main__':
    ads_path = sys.argv[1]
    changed_ads_path = sys.argv[2]
    deal_data(ads_path, changed_ads_path)
