import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats.stats import pearsonr
import numpy as np

def posted_contributors(df):
    df_postQ = df[(df['postTypeId'] == 1) & (df['userID'] != 'None')]
    df_dev = df_postQ.groupby('userID')

    total = 0
    groups = []
    for p in df_dev.groups:
        total += len(df_dev.groups[p])
        groups.append(len(df_dev.groups[p]))

    plt.hist(groups, bins=10)
    plt.title("The distribution of the contributors who post questions")
    plt.xlabel("Number of posted questions")
    plt.ylabel("Number of developers")
    # plt.show()
    plt.savefig('Distribution of Questioners.png', bbox_inches='tight')


def answer_contributors(df):
    df_ansQ = df[(df['postTypeId'] == 2) & (df['userID'] != 'None')]
    df_dev = df_ansQ.groupby('userID')

    total = 0
    groups = []
    for p in df_dev.groups:
        total += len(df_dev.groups[p])
        groups.append(len(df_dev.groups[p]))

    plt.hist(groups)
    plt.title("The distribution of the contributors who answer questions")
    plt.xlabel("Number of answered questions")
    plt.ylabel("Number of developers")
    # plt.show()
    plt.savefig('Distribution of Answerers.png', bbox_inches='tight')


def askQ_ansQ(df):
    df_postQ = df[(df['postTypeId'] == 1) & (df['userID'] != 'None')]
    df_ansQ = df[(df['postTypeId'] == 2) & (df['userID'] != 'None')]

    postQ = df_postQ['userID'].drop_duplicates().tolist()
    ansQ = df_ansQ['userID'].drop_duplicates().tolist()
    post_ans_Q = list(set(postQ) & set(ansQ))
    print 'Ratio: %f' % (len(post_ans_Q) / (float(len(postQ))))


def receivingHelp(df):
    df_postQ = df[(df['postTypeId'] == 1) & (df['userID'] != 'None')]
    df_ansQ = df[(df['postTypeId'] == 2) & (df['userID'] != 'None')]

    postQ = df_postQ['userID'].drop_duplicates().tolist()
    ansQ = df_ansQ['userID'].drop_duplicates().tolist()
    post_ans_Q = list(set(postQ) & set(ansQ))

    u_posts, u_ans = [], []
    for uID in post_ans_Q:
        postQ = df[(df['postTypeId'] == 1) & (df['userID'] == uID)]
        ansQ = df[(df['postTypeId'] == 2) & (df['userID'] == uID)]
        u_posts.append(len(postQ))
        u_ans.append(len(ansQ))
        # print len(postQ), len(ansQ)

    print 'Pair t-test: %s' % str(stats.ttest_rel(u_posts, u_ans))
    # print 'Correlation %f' % np.corrcoef(np.array(u_posts), np.array(u_ans))

if __name__ == '__main__':

    tree = ET.parse('dataset.xml')
    root = tree.getroot()
    ids, typeIds, parentIDs, userIDs = [], [], [], []

    for child in root.findall('row'):
        id_, postTypeId, parentID, userID = child.get('Id'), child.get('PostTypeId'), \
                                            child.get('ParentId'), child.get('OwnerUserId')
        ids.append(int(id_)), typeIds.append(int(postTypeId)), parentIDs.append(parentID), userIDs.append(str(userID))
    df_ = pd.DataFrame({'Id': ids, 'postTypeId': typeIds, 'parentID': parentIDs, 'userID': userIDs})

    posted_contributors(df_)
    answer_contributors(df_)
    askQ_ansQ(df_)
    receivingHelp(df_)
