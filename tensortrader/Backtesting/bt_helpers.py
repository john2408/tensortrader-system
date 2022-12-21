import numpy as np

def adj_ml_strategy(_input : np.array, 
                   v_barrier_minutes : int, 
                   verbose :int = 0) -> np.array:
    """
    Adjust trading signals based on different
    rules. 
    
    Args: 
        signals (np.array) : array of trading signals
        v_barrier_minutes (int): vertical barrier holding
                                    period
    
    Returns:
        Adjusted array of trading signals
        
    ## Unit Test
    v_barrier_minutes = 5
    _input = [-1, 1, 0, 1, 0, 0, 0, 0, 0, -1, 1, 1, 0, 0, 1, 1, 0 , 1]
    _output = [0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0 , 1]


    assert all(np.array(adj_pred) == np.array(_output)), "Error"
    """

    n_signals = len(_input)

    index = 0
    adj_pred = []

    while(index < n_signals):

        if _input[index] == 1:


            index += 1
            adj_pred.append(1)
            if index > n_signals -1:
                break
            if verbose > 1:
                print(index, ": old value ", _input[index], "new value", 1)

            for i in range(1, v_barrier_minutes):

                adj_pred.append(0)
                index += 1

                if index > n_signals -1:
                    break
                    
                if verbose > 1:
                    print(index, ": old value for", _input[index], "new value:", 0)


            if index < n_signals:
                adj_pred.append(-1)
                index += 1
                
                if verbose > 1:
                    print(index, ": old value for", _input[index], "new value:", -1)
            else:
                break

        # Otherwise add 0
        if index >= n_signals:
            break

        adj_pred.append(0)        
        index +=1
        
        if verbose > 1:
            print(index, ": old value", _input[index], "new value", 0)
        
    assert len(adj_pred) == len(_input), " Different input/ouptput lengths"
        
    return adj_pred