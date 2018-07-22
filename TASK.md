
# Problem Statement

Your client is an Insurance company and they need your help in building a model
to predict the propensity to pay renewal premium and build an incentive plan for
its agents to maximise the net revenue (i.e. renewals - incentives given to
collect the renewals) collected from the policies post their issuance.

 

You have information about past transactions from the policy holders along with
their demographics. The client has provided aggregated historical transactional
data like number of premiums delayed by 3/ 6/ 12 months across all the products,
number of premiums paid, customer sourcing channel and customer demographics
like age, monthly income and area type.

 

In addition to the information above, the client has provided the following
relationships:

  1. Expected effort in hours put in by an agent for incentives provided; and
  2. Expected increase in chances of renewal, given the effort from the agent.

 

Given the information, the client wants you to predict the propensity of renewal
collection and create an incentive plan for agents (at policy level) to maximise
the net revenues from these policies.



## Evaluation Criteria

Your solutions will be evaluated on 2 criteria:

  - (A) The base probability of receiving a premium on a policy without
        considering any incentive
  - (B) The monthly incentives you will provide on each policy to maximize the
        net revenue 


### Part A:

The probabilities predicted by the participants would be evaluated using AUC ROC
score.

 
### Part B:

The net revenue across all policies will be calculated in the following manner:

![$$\text{total net revenue} = \sum\limits_{\text{policies}} \left[(p_{\text{benchmark}} + \Delta p) * \text{premium} - \text{incentives}\right],$$ where:
](http://quicklatex.com/cache3/aa/ql_75b4bec0a748fe6a11baedc9c6747eaa_l3.png)

where:

  - ![$p_{\text{benchmark}}](http://quicklatex.com/cache3/db/ql_df8aa27db8fb3db852ca7513bd8b40db_l3.png)
    is the renewal probability predicted using a benchmark model by the
    insurance company
  - ![$\Delta p$](http://quicklatex.com/cache3/30/ql_32aa99d772b5af18666e68bf5d3c8f30_l3.png)
    (% improvement in renewal probability * ![](http://quicklatex.com/cache3/db/ql_df8aa27db8fb3db852ca7513bd8b40db_l3.png))
    is the improvement in renewal probability calculated from the agent efforts
    in  hours
  - 'Premium on policy' is the premium paid by the policy holder for the policy
    in consideration
  - 'Incentive on policy' is the incentive given to the agent for increasing the
    change of renewal (estimated by the participant) for each policy

The following curve provide the relationship between extra effort in hours
invested by the agent with Incentive to the agent and % improvement in renewal
probability vs agent effort in hours.

  1. Relationship b/w Extra efforts in hours invested by an agent and Incentive
     to agent. After a point more incentives does not convert to extra efforts.

        [plot]

     Equation for the effort-incentives curve:
     
     ![Y = 10*(1-exp(-X/400))](http://quicklatex.com/cache3/76/ql_aea10e2b887b700aa5c1801057022f76_l3.png)

  2. Relationship between % improvement in renewal probability vs Agent effort
     in hours. The renewal probability cannot be improved beyond a certain level
     even with more efforts.

        [plot]

     Equation for the % improvement in renewal prob vs effort curve:
     
     ![Y = 20*(1-exp(-X/5))](http://quicklatex.com/cache3/ca/ql_94727c2d99fc22774310ca2a13aec0ca_l3.png)

 
*Note: The client has used sophisticated psychological research to arrive at
these relationships and you can assume them to be true.*

Overall Ranking at the leaderboard would be done using the following equation:

![Combined Score = w1*AUC-ROC value + w2*(net revenue collected from all policies)*lambda](http://quicklatex.com/cache3/d8/ql_86522d798b2377338f328fc4210a7bd8_l3.png)

Where

  - ![w1 = 0.7](http://quicklatex.com/cache3/e4/ql_259162b01a6bd942ff99e3f3e60cdce4_l3.png)
  - ![w2 = 0.3](http://quicklatex.com/cache3/aa/ql_4da059e6ec6a7f9d70b7001dee60f2aa_l3.png)
  - lambda is a normalizing factor


### Public and Private Split:

Public leaderboard is based on 40% of the policies, while private leaderboard
will be evaluated on remaining 60% of policies in the test dataset.


### Data

#### `train.csv`

It contains training data for customers along with renewal premium status
(Renewed or Not?)

| Variable                         | Definition                                                                                                         |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------|
| id                               | Unique ID of the policy                                                                                            |
| perc_premium_paid_by_cash_credit | Percentage of premium amount paid by cash or credit card                                                           |
| age_in_days                      | Age in days of policy holder                                                                                       |
| Income                           | Monthly Income of policy holder                                                                                    |
| Count_3-6_months_late            | No of premiums late by 3 to 6 months                                                                               |
| Count_6-12_months_late           | No  of premiums late by 6 to 12 months                                                                             |
| Count_more_than_12_months_late   | No of premiums late by more than 12 months                                                                         |
| application_underwriting_score   | Underwriting Score of the applicant at the time of application (No applications under the score of 90 are insured) |
| no_of_premiums_paid              | Total premiums paid on time till now                                                                               |
| sourcing_channel                 | Sourcing channel for application                                                                                   |
| residence_area_type              | Area type of Residence (Urban/Rural)                                                                               |
| premium                          | Monthly premium amount                                                                                             |
| renewal                          | Policy Renewed? (0 - not renewed, 1 - renewed)                                                                     |

	
#### `test.csv`

Additionally test file contains premium which is required for the optimizing the
incentives for each policy in the test set.


| Variable                         | Definition                                                                                                         |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| id                               | Unique ID of the policy                                                                                            |
| perc_premium_paid_by_cash_credit | Percentage of premium amount paid by cash or credit card                                                           |
| age_in_days                      | Age in days of policy holder                                                                                       |
| Income                           | Monthly Income of policy holder                                                                                    |
| Count_3-6_months_late            | No of premiums late by 3 to 6 months                                                                               |
| Count_6-12_months_late           | No  of premiums late by 6 to 12 months                                                                             |
| Count_more_than_12_months_late   | No of premiums late by more than 12 months                                                                         |
| application_underwriting_score   | Underwriting Score of the applicant at the time of application (No applications under the score of 90 are insured) |
| no_of_premiums_paid              | Total premiums paid on time till now                                                                               |
| sourcing_channel                 | Sourcing channel for application                                                                                   |
| residence_area_type              | Area type of Residence (Urban/Rural)                                                                               |
| premium                          | Monthly premium amount                                                                                             |


#### `sample_submission.csv`

Please submit as per the given sample submission format only

| Variable   | Definition                     |
| ---------- | ------------------------------ |
| id         | Unique ID for the policy       |
| renewal    | Predicted Renewal Probability  |
| incentives | Incentives for agent on policy |
                                                                                          |
