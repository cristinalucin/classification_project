# Classification project

# Project Description:
TELCO is a telecommunications company that provides communication and media services to customers. Some of these services include: phone services, online security and backup, device protection, tech support, streaming tv, and streaming movies. The TELCO churn database has customer account information and customer demographics. I am looking into drivers of customer churn and seeking actionable conclusions to assist TELCO in preventing customer churn.

# Project Goal:
* Discover drivers of Telco customer churn
* Use drivers to develop machine learning models to predict whether or not a customer will churn.

# Initial Thoughts:
My initial hypothesis is that monthly charges and contract type are the biggest drivers of churn.

# The Plan
* Aquire data from codeup database

* Prepare data

* Explore data in search of drivers of churn

  * Answer the following initial questions:
    * Do senior citizens churn more or less?
    * Does a customer monthly charge rate affect churn?
    * Does contract type (month-to-month, one-year, two-year) impact customer churn?
    * Does having dependents impact customer churn?
  
* Develop a Model to predict if a customer will churn

    * Use drivers identified through exploration to build predictive models of different types
    * Evaluate models on train and validate data
    * Select the best model based on highest accuracy in combination with an examination of recall and precision
    * Evaluate the best model on test data
    
* Draw conclusions



# Data Dictionary

| Feature | Definition |
| --- | --- |
| Dependents | Whether the customer has dependents or not |
| Device Protection | Whether the customer has device protection (Yes, No, No internet service) |
| Gender | Whether a customer is male (0) or female (1) |
| Internet Service | Customer internet service type (DSL, Fiber optic, No internet service)|
| Monthly Charges | Amount paid by the customer per month |
| Phone Service | Whether the customer has a phone service or not |
| Multiple Lines | Whether the customer has multiple lines or not (Yes, No, No phone service)|
| Online Security | Whether the customer has online security or not (Yes, No, No internet service) |
| Online Backup | Whether the customer has online backup or not (Yes, No, No internet service) |
| Partner | Whether the customer has a partner or not |
| Payment Type | Whether the customer pays automatically (bank transfer or credit card) or manually
| Payment Type | (mailed check or electronic check)
| Senior Citizen | Whether a customer is Senior Citizen (1) or not (0) |
| Streaming TV | Whether the customer has streaming tv or not (Yes, No, No internet service) |
| Streaming Movies | Whether the customer has streaming movies or not (Yes, No, No internet service) |
| Tech Support | Whether the customer has tech support or not (Yes, No, No internet service) |
| Tenure | Number of months the customer has stayed with the company |
| Total Charges| Total charges accrued by the customer


| Additional Features | Encoded and values for categorical data and scaled versions continuous data|

# Steps to Reproduce

1. Clone this repository
2. Acquire the data from the TELCO churn database (located inside Codeup Database)
3. Put the data in the file containing the cloned repo.
4. Create or copy your env.py file to this repo, specifying the codeup hostname, username and password
5. Run notebook.

# Takeaways and Conclusions

* Exploration of the data revealed significant relationships between most features of this data and whether a customer would churn
* More time in exploration could yield a different combination of features, improving model accuracy.
* The combination of examining senior citizens, those with dependents, customer monthly charges, and customer contract type helped created a model with approximately 76 accuracy on training data. Its selection focused on replicability on testing data, which helped create a model that predicted accuracy within a half-percentage point of modeling on training data
* Besides accuracy, the chosen model maximized recall, which could further help Telco reduce customer churn

# Recommendations

* Focusing on feature selection or feature engineering to maximize future modeling accuracy
* Looking at features included in this model to target customers (Senior Citizens, those with dependents) with specific offers to reduce churn.

# Next Steps

* Feature engineering add-ons in different iterations
* Examining other possible drivers of churn based on customer feedback/sales reports