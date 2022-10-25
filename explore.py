

def cat_vis(train, col):
    '''
    This function takes in a categorical(cat) variable in the telco df
    and plots the univariate relationship with a barchart.
    '''
    plt.title('Relationship of churn rate and '+col)
    sns.barplot(x=col, y='churn_encoded', data=train)
    churn_rate = train.churn_encoded.mean()
    plt.axhline(churn_rate, label='churn rate')
    plt.legend()
    plt.show()


def cat_test(train, col):
    '''
    This function takes in training data and runs chi2 tests on
    categorical variables in relationship to the target, churn.
    '''
    alpha = 0.05
    null_hyp = col+' and churn rate are independent'
    alt_hyp = 'There is a relationship between churn rate and '+col
    observed = pd.crosstab(train.churn_encoded, train[col])
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    if p < alpha:
        print('We reject the null hypothesis that', null_hyp)
        print(alt_hyp)
    else:
        print('We fail to reject the null hypothesis that', null_hyp)
        print('There appears to be no relationship between survival rate and '+col)

def cat_analysis(train, col):
    cat_vis(train, col)
    cat_test(train, col)