import os
import requests
from tqdm import tqdm

DOWNLOAD_DIR = "D:\Deltacubes\python_programming\pdfs"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
pdf_links=[
'http://oar.icrisat.org/13523/1/Genetic%20Resources%20and%20Crop%20Evolution_72_673-696_2025.pdf',
'http://oar.icrisat.org/13129/1/Functional%20and%20Integrative%20Genomics_25_1-19_2025.pdf',
'http://oar.icrisat.org/13475/1/Genetic%20Resources%20and%20Crop%20Evolution_73-1-9_2025.pdf',
'http://oar.icrisat.org/13132/1/Frontiers%20in%20Plant%20Science_12_1-9_2021.pdf',
'http://oar.icrisat.org/13181/1/Scientific%20Reports_15_1-19_2025.pdf',
'http://oar.icrisat.org/12114/1/PLoS%20ONE_18_6_01-19_2023.pdf',
'http://oar.icrisat.org/12533/1/Electronic%20Journal%20of%20Plant%20Breeding_14_3_1147-1157_2023.pdf',
'http://oar.icrisat.org/12393/1/Crop%20Science_1-20_2023.pdf',
'http://oar.icrisat.org/12559/1/Agriculture_13_4_1-17_2023.pdf',
'http://oar.icrisat.org/12078/1/CABI%20Agriculture%20and%20Bioscience.pdf',
'http://oar.icrisat.org/12267/1/Scientific%20Reports_13_01-14_2023.pdf',
'http://oar.icrisat.org/12945/1/Crop%20Science_61_4_2658-2679_2021.pdf',
'http://oar.icrisat.org/12335/1/Key%20descriptors%20for%20fonio%20millets.pdf',
'http://oar.icrisat.org/12972/1/PNAS_118_38_1-9_2021.pdf',
'http://oar.icrisat.org/12203/1/Agriculture_13_2_01-14_2023.pdf',
'http://oar.icrisat.org/13044/1/Frontiers%20in%20Plant%20Science_15_1-19_2024.pdf',
'http://oar.icrisat.org/12373/1/Frontiers%20in%20Genetics_13_1-15_2022.pdf',
'http://oar.icrisat.org/12567/1/Euphytica_218_1-20_2022.pdf',
'http://oar.icrisat.org/12931/1/Frontiers%20in%20Sustainable%20Food%20Systems_8_01-15_2024.pdf',
'http://oar.icrisat.org/12688/1/Agronomy_14_1-11_2024.pdf',
'http://oar.icrisat.org/11678/1/plants-09-01296-v2.pdf',
'http://oar.icrisat.org/12422/1/Frontier%20Technologies%20Chapter%2002.pdf',
'http://oar.icrisat.org/13331/1/Agricultural%20Research%20Journal_61_2_239-247_2024.pdf',
'http://oar.icrisat.org/12356/1/Frontiers%20in%20Genetics_13_1-21_2022.pdf',
'http://oar.icrisat.org/11737/1/3546-Article%20Text-9861-1-10-20200715.pdf',
'http://oar.icrisat.org/12649/1/Key%20descriptors%20for%20foxtail%20millet.pdf',
'http://oar.icrisat.org/12385/1/Frontiers%20in%20Plant%20Science_14_01-15_2023.pdf',
'http://oar.icrisat.org/12724/1/Frontiers%20in%20Nutrition_11_1-11_2024.pdf',
'http://oar.icrisat.org/12066/1/Frontiers%20in%20Plant%20Science_13_1-15_2022.pdf',
'http://oar.icrisat.org/12809/1/Nutrients_14_1-16_2022.pdf',
'http://oar.icrisat.org/13492/1/Genetic%20Resources%20and%20Crop%20Evolution_73_1-16_2026.pdf',
'http://oar.icrisat.org/12468/1/1.%20Plant%20Stress_11_1-13_2024.pdf',
'http://oar.icrisat.org/13014/1/Plant%20Molecular%20Biology%20Reporter_43_180-196_2025.pdf',
'http://oar.icrisat.org/12443/1/Sustainability_14_1-17_2022.pdf',
'http://oar.icrisat.org/12507/1/Agriculture_13_4_1-17_2023.pdf',
'http://oar.icrisat.org/12176/1/Plant%20Physiology_191_1884-1912_2023.pdf',
'http://oar.icrisat.org/13241/1/BMC%20Plant%20Biology_24_1-14_2024.pdf',
'http://oar.icrisat.org/12212/1/BMC%20Plant%20Biology_23_1-17_2023.pdf',
'http://oar.icrisat.org/12407/1/Critical%20Reviews%20in%20Biotechnology_43_2_309-325_2023.pdf',
'http://oar.icrisat.org/12590/1/Frontiers%20in%20Plant%20Science_13_1-22_2022.pdf',
'http://oar.icrisat.org/12450/1/Nature%20Plants_8_491-499_2022.pdf',
'http://oar.icrisat.org/11728/1/fpls-11-587426.pdf',
'http://oar.icrisat.org/12769/1/Frontiers%20in%20Plant%20Science_12_1-14_2021.pdf',
'http://oar.icrisat.org/11739/1/3575-Article%20Text-10369-1-10-20201001.pdf',
'http://oar.icrisat.org/13169/1/Tropical%20Plant%20Biology_18_1-16_2025.pdf',
'http://oar.icrisat.org/12758/1/Indian%20Journal%20of%20Genetics%20and%20Plant%20Breeding_81_1_2021.pdf',
'http://oar.icrisat.org/13513/1/BMC%20Plant%20Biology_25_1-16_2025%20%283%29.pdf',
'http://oar.icrisat.org/12902/1/Agronomy_11_1-20_2021.pdf',
'http://oar.icrisat.org/13222/1/Plant%20Science%20Today_12_2_1-9_2025.pdf',
'http://oar.icrisat.org/12132/1/Nature_599_622-627_2021.pdf',
'http://oar.icrisat.org/12973/1/Journal%20of%20Experimental%20Botany_72_14_5158-5179_2021.pdf',
'http://oar.icrisat.org/12556/1/Frontiers%20in%20Plant%20Science_14_01-13_2023.pdf',
'http://oar.icrisat.org/12569/1/Molecular%20Biology%20Reports_49_5669-5683_2022.pdf',
'http://oar.icrisat.org/12633/1/The%20Plant%20Genome_1-16_2024.pdf',
'http://oar.icrisat.org/12571/1/Heredity_128_434-449_2022.pdf',
'http://oar.icrisat.org/12566/1/Rice_15_1-23_2022.pdf',
'http://oar.icrisat.org/11735/1/3464-Article%20Text-9709-1-10-20200623.pdf',
'http://oar.icrisat.org/13498/1/BMC%20Plant%20Biology_25_1-16_2025.pdf',
'http://oar.icrisat.org/12371/1/Journal%20of%20Experimental%20Botany_73_11_3584%E2%80%933596_2022.pdf',
'http://oar.icrisat.org/12976/1/Global%20Food%20Security_30_1-26_2021.pdf',
'http://oar.icrisat.org/12690/1/Plants_13_6_1-26_2024.pdf',
'http://oar.icrisat.org/12508/1/Plant%20Physiology_191_3_1884-1912_2023.pdf',
'http://oar.icrisat.org/13172/1/Molecular%20Genetics%20and%20Genomics_300_1-12_2025.pdf',
'http://oar.icrisat.org/12821/1/Scientific%20Reports_14_1-17_2024.pdf',
'http://oar.icrisat.org/13521/1/Cereal%20Research%20Communications_53_2047-2061_2025.pdf',
'http://oar.icrisat.org/13026/1/Genes_15_4_1-22_2024.pdf'
]

def download_pdfs(pdf_links):
    for url in tqdm(pdf_links):
        try:
            filename = url.split("/")[-1]
            filepath = os.path.join(DOWNLOAD_DIR, filename)

            # skip if already exists
            if os.path.exists(filepath):
                continue

            res = requests.get(url, stream=True, timeout=20)

            if res.status_code == 200:
                with open(filepath, "wb") as f:
                    for chunk in res.iter_content(1024):
                        f.write(chunk)

            print(f"Downloaded: {filename}")

        except Exception as e:
            print(f"Error: {url} -> {e}")

download_pdfs(pdf_links)

