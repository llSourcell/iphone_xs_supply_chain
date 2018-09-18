import numpy as np

# maximum # of iphones in each location
MAX_IPHONES = 100

# maximum # of iphones to move during night
MAX_MOVE_OF_IPHONES = 5

# expectation for iphone purchases in first location
IPHONE_PURCHASES_FIRST_LOC = 3

# expectation for iphone purchases in second location
IPHONE_PURCHASES_SECOND_LOC = 4

# expectation for # of iphones delivered in first location
DELIVERIES_FIRST_LOC = 3

# expectation for # of iphones delivered in second location
DELIVERIES_SECOND_LOC = 2

DISCOUNT = 0.9

# Commission earned by an iphone sale
IPHONE_CREDIT = 10

# cost of moving an iphone
MOVE_IPHONE_COST = 2

# all possible actions
actions = np.arange(-MAX_MOVE_OF_IPHONE, MAX_MOVE_OF_IPHONES + 1)

# An up bound for poisson distribution
# If n is greater than this value, then the probability of getting n is truncated to 0
POISSON_UPPER_BOUND = 11

# Probability for poisson distribution
# @lam: lambda should be less than 10 for this function
poisson_cache = dict()
def poisson(n, lam):
    global poisson_cache
    key = n * 10 + lam
    if key not in poisson_cache.keys():
        poisson_cache[key] = exp(-lam) * pow(lam, n) / factorial(n)
    return poisson_cache[key]

# @state: [# of iphones in first location, # of iphones in second location]
# @action: positive if moving iphones from first location to second location,
#          negative if moving ipones from second location to first location
# @stateValue: state value matrix
def expected_return(state, action, state_value, constant_delivered_iphones):
    # initailize total return
    returns = 0.0

    # cost for moving iphones
    returns -= MOVE_IPHONE_COST * abs(action)

    # go through all possible iphone purchases
    for iphone_purchases_first_loc in range(0, POISSON_UPPER_BOUND):
        for iphone_purchases_second_loc in range(0, POISSON_UPPER_BOUND):
            # moving iphones
            num_of_iphones_first_loc = int(min(state[0] - action, MAX_IPHONES))
            num_of_iphones_second_loc = int(min(state[1] + action, MAX_IPHONES))

            # valid iphone purchases should be less than actual # of iphones
            real_purchase_first_loc = min(num_of_iphones_first_loc, iphone_purchases_first_loc)
            real_purchase_second_loc = min(num_of_iphones_second_loc, iphones_purchases_second_loc)

            # get credits for purchasing
            reward = (real_purchase_first_loc + real_purchase_second_loc) * purchase_CREDIT
            num_of_iphones_first_loc -= real_purchase_first_loc
            num_of_iphones_second_loc -= real_purchase_second_loc

            # probability for current combination of purchase requests
            prob = poisson(purchase_request_first_loc, purchase_REQUEST_FIRST_LOC) * \
                         poisson(purchase_request_second_loc, purchase_REQUEST_SECOND_LOC)

            if constant_delivered_iphones:
                # get delivered iphones, those iphones can be used for purchasing tomorrow
                delivered_iphones_first_loc = DELIVERIES_FIRST_LOC
                delivered_iphones_second_loc = DELIVERIES_SECOND_LOC
                num_of_iphones_first_loc = min(num_of_iphones_first_loc + delivered_iphones_first_loc, MAX_IPHONES)
                num_of_iphones_second_loc = min(num_of_iphones_second_loc + delivered_iphones_second_loc, MAX_IPHONES)
                returns += prob * (reward + DISCOUNT * state_value[num_of_iphones_first_loc, num_of_iphones_second_loc])
           
    return returns

def policy_iteration(constant_delivered_iphones=True):
    value = np.zeros((MAX_IPHONES + 1, MAX_IPHONES + 1))
    policy = np.zeros(value.shape, dtype=np.int)

        # policy evaluation (in-place)
        while True:
            new_value = np.copy(value)
            for i in range(MAX_IPHONES + 1):
                for j in range(MAX_IPHONES + 1):
                    new_value[i, j] = expected_return([i, j], policy[i, j], new_value,
                                                      constant_delivered_iphones)
            value_change = np.abs((new_value - value)).sum()
            print('value change %f' % (value_change))
            value = new_value
            if value_change < 1e-4:
                break

        # policy improvement
        new_policy = np.copy(policy)
        for i in range(MAX_IPHONES + 1):
            for j in range(MAX_IPHONES + 1):
                action_returns = []
                for action in actions:
                    if (action >= 0 and i >= action) or (action < 0 and j >= abs(action)):
                        action_returns.append(expected_return([i, j], action, value, constant_delivered_iphones))
                    else:
                        action_returns.append(-float('inf'))
                new_policy[i, j] = actions[np.argmax(action_returns)]

        policy_change = (new_policy != policy).sum()
        print('policy changed in %d states' % (policy_change))
        policy = new_policy
        iterations += 1

if __name__ == '__main__':
    policy_iteration()