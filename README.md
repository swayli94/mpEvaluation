# mpEvaluation

Multi-process evaluation of a user-defined function `y=func(x, **kwargs)`.

```python
def func(x, **kwargs):
    '''
    >>> y = func(x, **kwargs)
    
    Parameters
    --------------
    x: ndarray [dim_input]
        function input
    kwargs: dict
        key word arguments
       
    Returns
    --------------
    y: ndarray [dim_output]
        function output
    '''
```

## Classes

```python
class MultiProcessEvaluation():
    '''
    Multi-process evaluation of a user-defined function `y=func(x, **kwargs)`.
    
    >>> mpRun = MultiProcessEvaluation(dim_input, dim_output, func=None, 
    >>>                 n_process=None, information=True, timeout=None)
    
    Parameters
    --------------
    dim_input: int
        dimension of the function input `x`
    dim_output: int
        dimension of the function input `y`
    func: callable or None
        the user-defined function. 
        If `func` is None, it uses an external evaluation script to get the result.
        The details are explained in function `external_run`.
    n_process: int or None
        maximum number of processors. If `n_process` is None, use serial computation.
    information: bool
        whether print information on screen
    timeout: float or None
        limit to the wait time. If `timeout` is None, no limit on wait time.
    '''

class Problem():
    '''
    Problem for the user-defined function.
    
    >>> prob = Problem(dim_input, dim_output, name_inputs=None, name_outputs=None)
    
    Parameters
    --------------
    dim_input: int
        dimension of the function input `x`
    dim_output: int
        dimension of the function input `y`
    name_inputs: list[str] or None
        names of the function input `x`.
        If `name_inputs` is None, the default name is `x*`.
    name_outputs: list[str] or None
        names of the function input `y`.
        If `name_outputs` is None, the default name is `y*`.
    '''
```

## Example

```python
def usr_func(x, **kwargs):
    return True, np.array([np.sum(x**2)])

if __name__ == '__main__':

    dim_input = 2
    dim_output = 1
    n_sample = 10
    n_process = 4
    
    xs = np.array([[i,i] for i in range(n_sample)])

    mpRun = MultiProcessEvaluation(dim_input, dim_output, func=usr_func, n_process=n_process)

    _, ys = mpRun.evaluate(xs, list_name)

    for i in range(min(4, n_sample)):
        print('%3d x= '%(i), xs[i,:], ' y= ', ys[i,:])
```

