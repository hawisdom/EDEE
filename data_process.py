import logging
import itertools
logger = logging.getLogger(__name__)

event_roles = {
    'EquityFreeze': ['B_EquityHolder', 'I_EquityHolder', 'B_FrozeShares', 'I_FrozeShares', 'B_LegalInstitution',
           'I_LegalInstitution', 'B_TotalHoldingShares', 'I_TotalHoldingShares', 'B_TotalHoldingRatio',
           'I_TotalHoldingRatio', 'B_StartDate', 'I_StartDate', 'B_EndDate', 'I_EndDate',
           'B_UnfrozeDate', 'I_UnfrozeDate'],
    'EquityRepurchase': ['B_CompanyName', 'I_CompanyName', 'B_HighestTradingPrice', 'I_HighestTradingPrice',
           'B_LowestTradingPrice', 'I_LowestTradingPrice', 'B_RepurchasedShares', 'I_RepurchasedShares',
           'B_ClosingDate', 'I_ClosingDate', 'B_RepurchaseAmount', 'I_RepurchaseAmount'],
    'EquityUnderweight': ['B_EquityHolder', 'I_EquityHolder', 'B_TradedShares', 'I_TradedShares', 'B_StartDate',
           'I_StartDate', 'B_EndDate', 'I_EndDate', 'B_LaterHoldingShares', 'I_LaterHoldingShares',
           'B_AveragePrice', 'I_AveragePrice'],
    'EquityOverweight': ['B_EquityHolder', 'I_EquityHolder', 'B_TradedShares', 'I_TradedShares', 'B_StartDate',
           'I_StartDate', 'B_EndDate', 'I_EndDate', 'B_LaterHoldingShares', 'I_LaterHoldingShares',
           'B_AveragePrice', 'I_AveragePrice'],
    'EquityPledge': ['B_Pledger', 'I_Pledger', 'B_PledgedShares', 'I_PledgedShares', 'B_Pledgee', 'I_Pledgee',
           'B_TotalHoldingShares', 'I_TotalHoldingShares', 'B_TotalHoldingRatio', 'I_TotalHoldingRatio',
           'B_TotalPledgedShares', 'I_TotalPledgedShares', 'B_StartDate', 'I_StartDate',
           'B_EndDate', 'I_EndDate', 'B_ReleasedDate', 'I_ReleasedDate']}

event_type2idx = {'EquityFreeze':0,'EquityRepurchase':1,'EquityUnderweight':2,'EquityOverweight':3,'EquityPledge':4}
idx2event_type = {0:'EquityFreeze',1:'EquityRepurchase',2:'EquityUnderweight',3:'EquityOverweight',4:'EquityPledge'}

def get_role_role2idx():
    all_role_roles = []
    for event_type,roles in event_roles.items():
        role_role_list = list(itertools.product(roles,repeat=2))
        for role_role in role_role_list:
            all_role_roles.append((event_type,role_role[0],role_role[1]))

    role_role2idx = {'non_conn':0}
    for idx,role_role in enumerate(all_role_roles):
        role_role2idx[role_role] = idx+1

    idx2role_role = {idx:role_role for role_role,idx in role_role2idx.items()}

    return role_role2idx,idx2role_role

def get_stop_words():
    stopwords = []
    with open('./data/stopwords.txt', 'r', encoding='utf-8') as f_stopword:
        stopword_datas = f_stopword.readlines()
        for stopword in stopword_datas:
            stopwords.append(stopword.strip())
    return stopwords