"""Script for generating the adversarial dev and test sets."""
from arct import data
import pandas as pd


test_claim_map = {
    'Comment sections have not failed': 'Comment sections have failed',
    'Google is not a harmful monopoly': 'Google is a harmful monopoly',
    'Supreme court justice can denounce a candidate': 'Supreme court justice cannot denounce a candidate',
    'Brazil should postpone Olympics': 'Brazil should not postpone Olympics',
    'Christians have created a harmful atmosphere for gays': 'Christians have not created a harmful atmosphere for gays',
    'Government should regulate airline industry again': 'Government should not regulate airline industry again',
    'Marijuana is not a gateway drug': 'Marijuana is a gateway drug',
    'Medical websites are healthful': 'Medical websites are healthful',
    'Guns should be permitted on college campuses': 'Guns should not be permitted on college campuses',
    "Jail record should be an employer's first impression": "Jail record should not be an employer's first impression",
    'Brazil should not postpone Olympics': 'Brazil should postpone Olympics',
    'Foreign language classes should be mandatory in college': 'Foreign language classes should not be mandatory in college',
    'Felons should be allowed to vote': 'Felons should not be allowed to vote',
    'Mother Teresa should not be canonized': 'Mother Teresa should be canonized',
    'The U.S. Embassy should be moved from Tel Aviv to Jerusalem': 'The U.S. Embassy should not be moved from Tel Aviv to Jerusalem',
    'Guns should not be permitted on college campuses': 'Guns should be permitted on college campuses',
    'Supreme court justice cannot denounce a candidate': 'Supreme court justice can denounce a candidate',
    'Non-Muslims hurt women by wearing hijabs': 'Non-Muslims do not hurt women by wearing hijabs',
    'Kurds are allies in Syria': 'Kurds are not allies in Syria',
    'Comment sections have failed': 'Comment sections have not failed',
    "Turkey doesn't belong to NATO": 'Turkey does belong to NATO',
    'Opioid training should be mandatory': 'Opioid training should not be mandatory',
    'The A.D.H.D. diagnosis is helping kids': 'The A.D.H.D. diagnosis is not helping kids',
    'Overcrowded national parks should have restricted access': 'Overcrowded national parks should not have restricted access',
    'Google is a harmful monopoly': 'Google is not a harmful monopoly',
    'Christians have not created a harmful atmosphere for gays': 'Christians have created a harmful atmosphere for gays',
    'The U.S. Embassy should not be moved from Tel Aviv to Jerusalem': 'The U.S. Embassy should be moved from Tel Aviv to Jerusalem',
    "College students' votes do matter in an election": "College students' votes do not matter in an election",
    "Jail record should not be an employer's first impression": "Jail record should be an employer's first impression",
    'Obamacare is sustainable': 'Obamacare is not sustainable',
    'College should not be free': 'College should be free',
    'Iran remains a threat': 'Iran does not remain a threat',
    'College should be free': 'College should not be free',
    "Medicare doesn't need to be reformed": "Medicare does need to be reformed",
    "Iran doesn't remain a threat": "Iran does remain a threat",
    'Public universities are neglecting in-state students': 'Public universities are not neglecting in-state students',
    'Medicare needs to be reformed': 'Medicare does not need to be reformed',
    'Foreign language classes should not be mandatory in college': 'Foreign language classes should be mandatory in college',
    'Obamacare is not sustainable': 'Obamacare is sustainable',
    'Facebook is ruining journalism': 'Facebook is not ruining journalism',
    'Activists cannot be politicians': 'Activists can be politicians',
    'The world is becoming safer': 'The world is not becoming safer',
    'Machines are not gaining the upper hand on humans': 'Machines are gaining the upper hand on humans',
    'Overcrowded national parks should not have restricted access': 'Overcrowded national parks should have restricted access',
    'Mother Teresa should be canonized': 'Mother Teresa should not be canonized'}
dev_claim_map = {
    'Police is too willing to use force': 'Police is not too willing to use force',
    'Immigration is really a problem': 'Immigration is not really a problem',
    'Pollings undermine democracy': 'Pollings do not undermine democracy',
    'Abolish Birthright Citizenship': 'Do not abolish Birthright Citizenship',
    'Supreme court is not too powerful': 'Supreme court is too powerful',
    'Overcrowding is not a legitimate threat': 'Overcrowding is a legitimate threat',
    "Companies can't be trusted": "Companies can be trusted",
    'Overcrowding is a legitimate threat': 'Overcrowding is not a legitimate threat',
    'It is not fair to rate professors online': 'It is fair to rate professors online',
    'Russia can be a partner': 'Russia cannot be a partner',
    'Women should be encouraged to have kids at home': 'Women should not be encouraged to have kids at home',
    'Same-sex colleges are outdated': 'Same-sex colleges are not outdated',
    'It should be illegal to declaw your cat': 'It should not be illegal to declaw your cat',
    'Apps cannot be used to treat depression and anxiety': 'Apps can be used to treat depression and anxiety',
    'Same-sex colleges are still relevant': 'Same-sex colleges are not still relevant',
    "Parents' religious beliefs should not allow them to refuse medical care for their children": "Parents' religious beliefs should allow them to refuse medical care for their children",
    'It should be legal to declaw your cat': 'It should not be legal to declaw your cat',
    'Former colonial powers should not pay reparations': 'Former colonial powers should pay reparations',
    'Drug addicts should not be forced into treatment': 'Drug addicts be forced into treatment',
    'It is fair to rate professors online': 'It is not fair to rate professors online',
    'Open office layout should be reconsidered': 'Open office layout should not be reconsidered',
    'The Atlantic coast should not be opened to drilling': 'The Atlantic coast should be opened to drilling',
    'Drug addicts should be forced into treatment': 'Drug addicts should not be forced into treatment',
    'Economists are overrated': 'Economists are not overrated',
    'Russia cannot be a partner': 'Russia can be a partner',
    'Economists are not overrated': 'Economists are overrated',
    'Europe should shun refugees': 'Europe should not shun refugees',
    'Helping condo developers hurt the city': 'Helping condo developers help the city',
    'Internet addiction is not a concern for teenagers': 'Internet addiction is a concern for teenagers',
    'Companies can excel without making workers miserable': 'Companies cannot excel without making workers miserable',
    'The bar is too low to get into law school': 'The bar is not too low to get into law school',
    'Women should not be encouraged to have kids at home': 'Women should be encouraged to have kids at home',
    'Legitimize Frisbee': 'Do not legitimize Frisbee',
    'The threat has been ignored': 'The threat has not been ignored',
    'Immigration is not really a problem': 'Immigration is really a problem',
    'The threat has been exaggerated': 'The threat has not been exaggerated',
    'Helping condo developers help the city': 'Helping condo developers hurt the city',
    "Greece's anti-austerity government can't succeed": "Greece's anti-austerity government can succeed",
    "Parents' religious beliefs should allow them to refuse medical care for their children": "Parents' religious beliefs should not allow them to refuse medical care for their children",
    'Technological innovation is creating better worlds': 'Technological innovation is not creating better worlds',
    'Happy hour should not be banned': 'Happy hour should be banned',
    'Greece should not abandon Euro': 'Greece should abandon Euro',
    'Keep Birthright Citizenship': 'Do not keep Birthright Citizenship',
    'Fraternities and sororities should be co-ed': 'Fraternities and sororities should not be co-ed',
    'The bar is not too low to get into law school': 'The bar is too low to get into law school',
    'Supreme court is too powerful': 'Supreme court is not too powerful',
    'Do not legitimize Frisbee': 'Legitimize Frisbee',
    'Europe should not shun refugees': 'Europe should shun refugees',
    'Police is reacting to dangers': 'Police is not reacting to dangers',
    'Greece should abandon Euro': 'Greece should not abandon the Euro'}


def make(dataset):
    claim_map = dev_claim_map if dataset == 'dev' else test_claim_map
    df = data.load(dataset)
    file_name = '%s-adv-full.txt' % dataset
    new_claims = [claim_map[c] for c in list(df.claim)]
    new_labels = [not l for l in df.correctLabelW0orW1]
    adv = df.copy()
    adv.claim = new_claims
    adv.correctLabelW0orW1 = new_labels
    adv.to_csv('data/arct/%s' % file_name, sep='\t', index=False)


def merge(dataset):
    dfo = data.load(dataset)
    dfa = data.load('%s-adv' % dataset)
    dfm = pd.concat([dfo, dfa])
    dfm.to_csv('data/arct/%s.csv' % dataset, sep='\t', index=False)


if __name__ == '__main__':
    make('dev')
    make('test')
    merge('dev')
    merge('test')
