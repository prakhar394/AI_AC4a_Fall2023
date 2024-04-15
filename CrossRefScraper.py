import requests
import xml.etree.ElementTree as ET

def fetch_metadata(doi):
    """Fetch metadata for a given DOI."""
    headers = {'Accept': 'application/vnd.crossref.unixsd+xml'}
    url = f'http://dx.doi.org/{doi}'
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # This will raise an HTTPError if the request returned an unsuccessful status code
        return response.text
    except requests.HTTPError as e:
        print(f"HTTP Error: {e.response.status_code} - {e.response.reason}")
    except requests.RequestException as e:
        print(f"Request Error: {e}")
    return None


def parse_links_and_license(metadata):
    try:
        # Register namespace if any, to handle namespaced XML elements
        ET.register_namespace('', 'http://www.crossref.org/schema/1.1')
        root = ET.fromstring(metadata)

        # Initialize containers for links and license references
        links = []
        license_refs = []

        # Look for fulltext links in the standard locations
        for link in root.findall(".//{*}link"):
            if 'fulltext' in link.get('rel', ''):
                links.append(link.get('href'))

        # Also look for links under text-mining collections
        for resource in root.findall(".//{*}collection[@property='text-mining']/{*}item/{*}resource"):
            if resource.get('mime_type') == 'application/pdf':
                links.append(resource.text)

        # Look for license references
        for license_ref in root.findall(".//{*}license_ref"):
            license_refs.append(license_ref.text)

        return links, license_refs
    except ET.ParseError as e:
        print(f"XML Parse Error: {e}\nProblematic metadata: {metadata}")
        return [], []

def check_license(license_refs, safelist):
    """Check if any of the license references are in the safelist."""
    for license_ref in license_refs:
        if license_ref in safelist:
            return True
    return False

def main(doi_list, safelist):
    all_full_text_links = []  # List to store all links

    for doi in doi_list:
        print(f"Processing DOI: {doi}")
        metadata = fetch_metadata(doi)
        if metadata:
            links, license_refs = parse_links_and_license(metadata)
            if check_license(license_refs, safelist):
                if links:
                    all_full_text_links.extend(links)  # Add the links to the list
                else:
                    print("No full-text links available for DOI:", doi)
            else:
                print("License not in safelist for DOI:", license_refs)
        else:
            print("Failed to fetch or parse metadata for DOI:", doi)

    return all_full_text_links

# Example usage with multiple DOIs
safelist = ['http://creativecommons.org/licenses/by/3.0/deed.en_US', 'http://onlinelibrary.wiley.com/termsAndConditions#vor', 'http://journals.sagepub.com/page/policies/text-and-data-mining-license', 'http://www.springer.com/tdm','https://creativecommons.org/licenses/by-nc/4.0/', 'http://doi.wiley.com/10.1002/tdm_license_1.1', 'https://www.springernature.com/gp/researchers/text-and-data-mining', 'https://www.cambridge.org/core/terms', '']
doi_list = ['10.1002/ejsp.444', '10.1177/0010414012463884', '10.1207/S15327957PSPR0704_03', '10.1037/a0011989', '10.1111/j.1744-6570.2005.00633.x', '10.1016/j.appdev.2013.11.002', '10.1177/1368430214542256', '10.1111/jasp.12449', '10.1002/ejsp.504', '10.1037/pac0000243', '10.1037/0022-3514.90.5.751', '10.1016/j.ijintrel.2011.03.001', '10.1177/0146167216689055', '10.1177/0146167209337039', '10.1111/j.1559-1816.2011.00752.x', '10.4119/UNIBI/ijcv.61', '10.1207/s15327949pac1102_4', '10.1111/j.1467-9221.2008.00634.x', '10.1002/ejsp.409', '10.1016/j.ijintrel.2011.12.009', '10.1016/j.cpr.2017.08.006', '10.1080/00958961003676314', '10.1080/17400201.2013.862920', '10.1017/S1755773912000264', '10.1177/0192512110367663', '10.1016/j.jesp.2007.04.008', '10.1177/1088868314530518', '10.1207/s15327949pac0802_3', '10.1163/156853708X358194', '10.1177/0022002715607611', '10.1177/1948550613490967', '10.1177/0022022105275962', '10.1080/15534511003783521', '10.1177/0022002711429309', '10.1002/jclp.20237', '10.1037/a0030939', '10.1002/per.526', '10.1037/a0028620', '10.1111/pops.12589', '10.1177/0963721412471346', '10.1111/pops.12582', '10.1126/science.1202925', '10.1177/0022022108328919', '10.1177/0146167209337037', '10.17105/spr-2017-0083.v47-1', '10.1177/0146167213505872', '10.1177/0146167208325004', '10.1093/ijtj/ijq005', '10.1093/ijtj/ijl008', '10.1111/j.1460-2466.1999.tb02784.x', '10.1177/0261927X10387105', '10.1093/ijtj/ijm005', '10.1016/j.ijintrel.2013.09.006', '10.1177/1368430214542257', '10.1177/0022343304041068', '10.1177/0146167209337034', '10.1207/S15327957PSPR0602_01', '10.1177/1368430208098778', '10.1017/S0022381614000103', '10.1111/j.2044-8309.1991.tb00943.x', '10.1017/CBO9780511756252', '10.1002/9781118367377.ch12', '10.1111/1467-6494.7106007', '10.1002/9781118367377.ch12', '10.1016/S0191-8869(99)00135-X', '10.1007/s12552-015-9150-9', '10.1002/ejsp.2189', '10.1016/S1534-0856(02)04005-7', '10.1037/pac0000216', '10.1207/s15324834basp2701_2', '10.1016/j.ijintrel.2017.04.014', '10.1016/S1359-1789(01)00045-3', '10.1016/j.ijintrel.2014.02.002', '10.1177/1368430215586604', '10.1086/507139', '10.1348/014466605X27162', '10.1017/CBO9780511756252', '10.1111/j.1540-4560.2009.01607.x', '10.1021/ed077p116', '10.1016/j.compedu.2010.10.010', '10.1037/0033-2909.89.1.47', '10.1177/0022343304041060', '10.1515/1935-1682.2917', '10.1017/s0376892900036663', '10.1080/02255189.2012.693049', '10.1080/02255189.2012.693049', '10.1016/j.cosust.2017.02.001', '10.1016/j.cosust.2017.02.001', '10.1016/j.jenvp.2011.04.001', '10.1016/j.jesp.2007.04.006', '10.1111/1467-9280.00262', '10.1037//0022-3514.82.3.359', '10.1177/0146167207304788', '10.1177/0022002714564427', '10.1017/CBO9780511750946.014', '10.1177/1368430215586604', '10.1016/j.obhdp.2008.02.012', '10.1016/j.obhdp.2008.06.005', '10.1126/science.1108062', '10.1037/a0023752', '10.1111/jasp.12019', '10.1177/1948550613484499', '10.1111/josi.12174', '10.1007/978-3-319-10687-8_6', '10.1037/0022-3514.59.1.17', '10.1037/0021-843X.101.2.293', '10.1037/0021-843X.101.2.314', '10.1177/0022002715569772', '10.1016/S0065-2601(08)60281-6', '10.1007/978-1-4614-3555-6_12', '10.1037/0022-3514.56.1.5', '10.1177/1745691614527464', '10.1207/s15327957pspr1004_4', '10.1093/oxfordhb/9780199270125.003.0017', '10.1177/0146167205276431', '10.1177/0956797610391102', '10.1111/pops.12029', '10.2307/2111684', '10.1007/s11109-005-4803-9', '10.1177/1368430207071345', '10.1111/j.1467-9221.2010.00767.x', '10.1037/0033-295X.113.1.84', '10.1207/S15324834BASP2504_4', '10.1177/0022002715569772', '10.1177/1948550613484499', '10.1111/asap.12023', '10.1111/jasp.12446', '10.1080/00224545.2017.1412933', '10.1111/bjso.12284', '10.1002/cbm.2014', '10.1163/156853708X358182', '10.1111/j.2044-8309.1991.tb00943.x', '10.2466/pr0.1998.83.3f.1395', '10.1037/h0035588', '10.1177/0956797612440102', '10.1111/j.1467-9280.2008.02261.x', '10.1037/0022-3514.69.3.437', '10.1007/s00127-009-0117-2', '10.1037/0022-3514.78.5.889', '10.1590/0101-3173.2021.V44N2.24.P345', '10.1111/j.1741-3737.2001.01185.x', '10.1016/S0065-2601(08)60382-2', '10.1002/per.623', '10.1177/1541344604270863', '10.1177/0146167209337037', '10.1177/13684302211019479', '10.1177/1368430208098778', '10.1016/j.avb.2003.09.001', '10.1007/s11205-009-9563-1', '10.1174/021347410790193504', '10.1093/ijtj/ijm003']
list = main(doi_list, safelist)
print(list)