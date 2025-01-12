'''
Multi-process evaluation of a user-defined function

ProcessPoolExecutor:

    https://docs.python.org/3/library/concurrent.futures.html

'''
import os
import numpy as np
import time
from typing import List, Tuple
import concurrent
import concurrent.futures
from concurrent.futures import as_completed
import platform


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

    def __init__(self, dim_input: int, dim_output: int, name_inputs=None, name_outputs=None) -> None:
        
        self.dim_input = dim_input
        self.dim_output = dim_output
        
        if name_inputs is None:
            self.name_inputs = ['x%d'%(i) for i in range(dim_input)]
            
        elif len(name_inputs) == dim_input:
            self.name_inputs = name_inputs
            
        else:
            raise Exception
        
        if name_outputs is None:
            self.name_outputs = ['y%d'%(i) for i in range(dim_output)]
            
        elif len(name_outputs) == dim_output:
            self.name_outputs = name_outputs
            
        else:
            raise Exception

        self.input_fname = 'input.txt'
        self.output_fname = 'output.txt'

    def external_run(self, name: str, x: np.ndarray, information=True, bash_name='run', timeout=None):
        '''
        External calculation by calling run.bat/.sh.
        
        Parameters
        -----------------
        name: str
            name of the current running folder, the working folder is ./Calculation/name.
        x: ndarray [dim_input]
            function input
        information: bool
            whether print information on screen
        bash_name: str
            name of the external running script, the default is 'run'.
        timeout: float, or None
            if `timeout` is None, waits for the application to end.
            If `timeout` is a float, wait for `timeout` seconds.

        Returns
        ----------------
        succeed: bool
            whether the evaluation succeed or not
        y: ndarray [dim_output]
            function output

        I/O files
        ----------------
        input_fname: str
            name of the file that contains information of `x`.
            Each line contains the name and value of one variable, e.g. 'x1   1.0'.
        input_fname: str
            name of the file that contains information of `y`.
            Each line contains the name and value of one variable, e.g. 'y1   1.0'.
        
        '''

        if platform.system() in 'Windows':
            folder = '.\\Calculation\\'+name
            out_name = folder+'\\'+self.output_fname
            in_name  = folder+'\\'+self.input_fname

            if not os.path.exists(in_name):
                os.system('xcopy /s /y  .\\Runfiles  '+folder+'\\  > nul')

                self.write_input(in_name, x)
                
                if isinstance(timeout, int) or isinstance(timeout, float):
                    os.system('start /min /d   '+folder+'  %s.bat'%(bash_name))
                    time.sleep(float(timeout))
                    
                else:
                    os.system('start /wait /min /d   '+folder+'  %s.bat'%(bash_name))
                    os.system('del   '+folder+'\\%s.bat'%(bash_name))

        else:
            folder = './Calculation/'+name
            out_name = folder+'/'+self.output_fname
            in_name  = folder+'/'+self.input_fname

            if not os.path.exists(in_name):
                #* Note: check input.txt because if the folder exists,
                #* command 'cp' will copy the Runfiles folder inside the targeted folder
                #* instead of overwrite it.
                os.system('cp -rf  ./Runfiles  '+folder+'/ ')

                self.write_input(in_name, x)

                os.system('cd '+folder+' &&  sh ./%s.sh >/dev/null'%(bash_name))
                os.system('rm -f '+folder+'/%s.sh'%(bash_name))

        #* Process results 
        succeed, y = self.read_output(out_name)

        if information and not succeed:
            print('    warning: [external_run] failed: %s'%(name))

        return succeed, y

    def write_input(self, fname: str, x: np.ndarray):
        '''
        Write x into fname. (each line: var_name, value)
        '''
        f = open(fname, 'w', encoding='utf-8')
        for i in range(x.shape[0]):
            f.write('  %20s  %20.9f \n'%(self.name_inputs[i], x[i]))
        f.close()
        
    def read_input(self, fname: str):
        '''
        Read input file [fname], (each line: var_name, value)

        Returns
        -------------
        succeed: bool
            whether the evaluation succeed or not
        x: ndarray [dim_input]
            function input
        '''
        
        succeed = True
        x = np.ones(self.n_var)

        if not os.path.exists(fname):
            return False, x

        f = open(fname, 'r+', encoding='utf-8')
        lines = f.readlines()
        
        if len(lines) == 0:
            return False, x
        
        dict_out = dict()
        for line in lines:
            line = line.split()
            dict_out[line[0]] = float(line[1])

        for i in range(self.dim_input):
            name_var = self.name_inputs[i]
            if not name_var in dict_out.keys():
                print('  Error: input [%s] is not in %s'%(name_var, fname))
                succeed = False
                continue
            x[i] = dict_out[name_var]

        return succeed, x

    def read_output(self, fname: str):
        '''
        Read output file [fname], (each line: var_name, value)

        Returns
        -------------
        succeed: bool
            whether the evaluation succeed or not
        y: ndarray [dim_input]
            function output
        '''
        
        succeed = True
        y = np.ones(self.dim_output)

        if not os.path.exists(fname):
            return False, y

        f = open(fname, 'r+', encoding='utf-8')
        lines = f.readlines()
        
        if len(lines) == 0:
            return False, y
        
        dict_out = dict()
        for line in lines:
            line = line.split()
            if len(line)==0 and len(dict_out)>0:
                break
            dict_out[line[0]] = float(line[1])

        for i in range(self.dim_output):
            name_out = self.name_outputs[i]
            if not name_out in dict_out.keys():
                print('  Error: output [%s] is not in %s'%(name_out, fname))
                succeed = False
                continue
            y[i] = dict_out[name_out]

        return succeed, y

    def output_database(self, fname: str, xs: np.ndarray, ids=None, ys=None, list_succeed=None):
        '''
        Output database, including the inputs and the outputs.
        
        Parameters
        ---------------
        fname: str
            database file name
            
        xs: ndarray [n, dim_input]
            function inputs
        
        ids: None, or list [n]
            list of sample ID
        
        ys: None, or ndarray [n, dim_output]
            function outputs
            
        list_succeed: None, or list [bool]
            list of succeed for each input.
            If it is provided, only output samples that succeeded its evaluation.
        '''
        if ids is None:
            ids = [i+1 for i in range(xs.shape[0])]
            
        with open(fname, 'w') as f:
            
            f.write('Variables= ID')
            
            if ys is not None:
                for j in range(self.dim_output):
                    f.write(' %14s'%(self.name_outputs[j]))
                
            for j in range(self.dim_input):
                f.write(' %14s'%(self.name_inputs[j]))
                    
            f.write('\n')
            
            for i in range(xs.shape[0]):
                
                if list_succeed is not None:
                    if not list_succeed[i]:
                        continue
                    
                f.write('   %10d'%(ids[i]))
                
                if ys is not None:
                    for j in range(self.dim_output):
                        f.write(' %14.6E'%(ys[i,j]))
                    
                for j in range(self.dim_input):
                    f.write(' %14.6E'%(xs[i,j]))
                f.write('\n')

    def read_database(self, fname: str, have_output=True, dim_input=None, dim_output=None):
        '''
        Read database from file.
        
        Parameters
        ----------------
        fname: str
            database file name
            
        have_output: bool
            whether the file contains sample output.
        
        dim_input: None, or int
            user specified input dimension.
            This is needed when load database from another problem.
        
        dim_output: None, or int
            user specified output dimension.
            This is needed when load database from another problem.
            
        Returns
        ----------------
        ids: list [n]
            list of sample ID
        
        xs: ndarray [n, dim_input]
            function inputs
        
        ys: None, or ndarray [n, dim_output]
            function outputs
        '''
        if dim_input is None:
            dim_input = self.dim_input
        
        if dim_output is None:
            dim_output = self.dim_output
        
        with open(fname, 'r') as f:
            lines = f.readlines()
            
        ids = []
        xs = []
        ys = [] if have_output else None
        
        for line in lines:
            
            line = line.split()
            
            if line[0]=='Variables=' or line[0]=='#':
                continue
            
            ids.append(int(line[0]))
            
            if have_output:
                
                ys.append([float(line[i+1]) for i in range(dim_output)])
            
                xs.append([float(line[i+1+dim_output]) for i in range(dim_input)])
                
            else:
                
                xs.append([float(line[i+1]) for i in range(dim_input)])
        
        if have_output:
            ys = np.array(ys)
        
        return ids, np.array(xs), ys
    
    def remove_duplicate_samples(self, xs: np.ndarray, ys: np.ndarray, ids: List[int], 
                                    indexes_parameter: List[int] | None = None) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
        '''
        Remove duplicate samples from the dataset.
        
        Parameters
        ----------------
        xs: ndarray [n, dim_input]
            function inputs
        
        ys: ndarray [n, dim_output]
            function outputs
        
        ids: list [n]
            list of sample ID
        
        indexes_parameter: list [n_parameter] or None
            list of indexes of the parameters that are used to determine the uniqueness of the samples.
            If `indexes_parameter` is None, all parameters are used.
        
        Returns
        ----------------
        xs_new: ndarray [n_new, dim_input]
            function inputs
            
        ys_new: ndarray [n_new, dim_output]
            function outputs
            
        ids_new: list [n_new]
            list of sample ID
        
        indexes: list [n]
            list of indexes of the unique samples in the original dataset.
        '''
        xs_for_check = xs if indexes_parameter is None else xs[:,indexes_parameter]
        
        indexes = []
        for i in range(xs.shape[0]):
            for j in range(i+1, xs.shape[0]):
                if np.all(xs_for_check[i,:] == xs_for_check[j,:]):
                    break
            else:
                indexes.append(i)
                
        xs_new = xs[indexes,:]
        ys_new = ys[indexes,:]
        ids_new = [ids[i] for i in indexes]
        
        return xs_new, ys_new, ids_new, indexes
    

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
        
    Notes
    ----------------
    The `if __name__ == '__main__'` is necessary for python multiprocessing.
    
    https://docs.python.org/3/library/multiprocessing.html
    
    For an explanation of why the `if __name__ == '__main__'` part is necessary, see Programming guidelines.
    
    https://docs.python.org/3/library/multiprocessing.html#multiprocessing-programming
    
    
    User-defined function:
    
    >>> succeed, y = func(x, **kwargs)
    >>> # x: ndarray [dim_input]
    >>> # y: ndarray [dim_output]
    >>> # succeed: bool

    Evaluation of `n` inputs:

    >>> list_succeed, ys = Multiprocessing.evaluate(xs, **kwargs)
    >>> # xs: ndarray [n, dim_input]
    >>> # ys: ndarray [n, dim_output]
    >>> # list_succeed: list [bool], length is n
    '''
    def __init__(self, dim_input: int, dim_output: int, func=None, 
                    n_process=None, information=True, timeout=None):
        '''
        Using concurrent.futures.ProcessPoolExecutor as executor

        Using submit to schedule the callable, fn, to be executed as 
        fn(*args **kwargs) and returns a Future object 
        representing the execution of the callable.

        >>> executor =  ProcessPoolExecutor(max_workers=None,
                    mp_context=None, initializer=None, initargs=())

        >>> future = executor.submit(fn, *args, **kwargs)

        Args:
        ---
        max_workers:    The maximum number of processes that can be used to execute the given calls. 
                        If None or not given then as many worker processes will be created as the machine has processors.
        mp_context:     A multiprocessing context to launch the workers. 
                        This object should provide SimpleQueue, Queue and Process.
        initializer:    A callable used to initialize worker processes.
        initargs:       A tuple of arguments to pass to the initializer.
        '''
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.func = func
        self.n_process = n_process
        self.information = information
        self.timeout = timeout

        if self.func is None:
            self.external_run(None, None, None, initialize=True)
        
    def external_run(self, name: str, x: np.ndarray, prob: Problem, initialize=False):
        '''
        External calculation by calling run.bat/.sh.
        
        >>> succeed, y = external_run(self, name, x, prob)
        
        Parameters
        -----------------
        name: str
            name of the current running folder, the working folder is ./Calculation/name.
        x: ndarray [dim_input]
            function input
        information: bool
            whether print information on screen
        bash_name: str
            name of the external running script, the default is 'run'.
        timeout: float, or None
            if `timeout` is None, waits for the application to end.
            If `timeout` is a float, wait for `timeout` seconds.

        Returns
        ----------------
        succeed: bool
            whether the evaluation succeed or not
        y: ndarray [dim_output]
            function output
        '''
        #* Initial check of the external running environment
        if initialize:
            
            if not os.path.exists('./Runfiles'):
                raise Exception('The folder [Runfiles] that contains the external evaluation script does not exist')
            
            if not os.path.exists('./Runfiles/run.bat') and not os.path.exists('./Runfiles/run.sh'):
                raise Exception('The external evaluation script [run.bat/.sh] does not exist')
            
            if not os.path.exists('./Calculation'):
                os.mkdir('./Calculation')

        else:
            #* External evaluation by prob
            succeed, y = prob.external_run(name, x, information=self.information, timeout=self.timeout)

            return succeed, y

    def func_mp(self, x: np.ndarray, i: int, **kwargs):
        '''
        Callable function for the ProcessPoolExecutor
        
        >>> succeed, y, i = func_mp(self, x, i, **kwargs)

        Parameters
        -----------------
        x: ndarray [dim_input]
            function input
        i: int
            index of this `x` in xs[n, dim_input]
        name: str
            name of the current running folder, the working folder is ./Calculation/name
        prob: Problem
            the problem for external runs

        Returns
        ----------------
        succeed: bool
            whether the evaluation succeed or not
        y: ndarray [dim_output]
            function output
        i: int
            index of this `x` in xs[n, dim_input]
        '''
        if self.func is None:
            
            if 'name' in kwargs.keys():
                name = kwargs['name']
            else:
                raise Exception('Must define name as the the working folder')

            if 'prob' in kwargs.keys():
                prob = kwargs['prob']
            else:
                raise Exception('Must provide Problem object `prob` for external running')

            succeed, y = self.external_run(name, x, prob)

        else:

            succeed, y = self.func(x, **kwargs)

        return succeed, y, i

    def evaluate(self, xs: np.ndarray, list_name=None, **kwargs):
        '''
        Evaluation of the multiple inputs `xs`.
        
        >>> list_succeed, ys = evaluate(xs, list_name)
        
        Parameters
        -----------------
        xs: ndarray [n, dim_input]
            function input
        list_name: list or None
            list of working folder names for external runs
        prob: Problem
            the problem for external runs
        n_show: int
            print number of succeed runs each n_show succeed runs
        
        Returns
        -----------------
        list_succeed: list [bool]
            list of succeed for each input
        ys: ndarray [n, dim_output]
            function output

        Notes
        -----------------
        Schedule the callable functions to be executed

        >>> future = executor.submit(fn, *args, **kwargs)

        returns a Future object representing the execution of the callable

        Yield futures as they complete (finished or cancelled)

        >>> for f in as_completed(futures, timeout=None):
        >>>     f.result()

        Any futures that completed before as_completed() is called will be yielded first. 
        The returned iterator raises a concurrent.futures.TimeoutError 
        if __next__() is called and the result isn't available after timeout seconds 
        from the original call to as_completed(). timeout can be an int or float. 
        If timeout is not specified or None, there is no limit to the wait time.
        #! This timeout will raise an Error
        '''
        n = xs.shape[0]
        ys = np.zeros([n, self.dim_output])
        list_succeed = [False for _ in range(n)]

        if self.func is None and not isinstance(list_name, list):
            raise Exception('Must provide a list of working folder names')

        n_show = 100
        if 'n_show' in kwargs.keys():
            n_show = kwargs['n_show']

        if 'prob' in kwargs.keys():
            prob = kwargs['prob']
        elif self.func is None:
            raise Exception('Must provide Problem object `prob` for external running')

        #* Serial calculation
        if self.n_process==None:

            if self.func is None:
                for i in range(n):
                    list_succeed[i], ys[i,:] = self.external_run(list_name[i], xs[i,:], prob)
            
            else:
                for i in range(n):
                    list_succeed[i], ys[i,:] = self.func(xs[i,:], **kwargs)

        #* Multiprocessing calculation
        else:

            with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_process) as executor:

                futures = []
                
                for i in range(n):

                    if self.func is None:
                        futures.append(executor.submit(self.func_mp, xs[i,:], i, name=list_name[i], **kwargs))
                        
                    else:
                        futures.append(executor.submit(self.func_mp, xs[i,:], i, **kwargs))

                num = 0
                t0 = time.perf_counter()
                        
                for f in as_completed(futures, timeout=self.timeout):

                    succeed, y, i = f.result()

                    ys[i,:] = y
                    list_succeed[i] = succeed
                    
                    if succeed:
                        num += 1
                        if num%n_show==0:
                            t1 = time.perf_counter()
                            print('  > parallel calculation done: n = %d, t = %.2f min'%(num, (t1-t0)/60.0))
                            
                self.executor = executor

        return list_succeed, ys

    def finalize(self):
        '''
        Signal the executor that it should free any resources.

        Calls to Executor.submit() and Executor.map() made 
        after shutdown will raise RuntimeError.

        >>> executor.shutdown(wait=True)
        '''
        self.executor.shutdown(wait=True)


def usr_func(x, **kwargs):
    return True, np.array([np.sum(x**2)])


#* Example
if __name__ == '__main__':

    dim_input = 2
    dim_output = 1
    n_sample = 10
    n_process = 4
    
    prob = Problem(dim_input, dim_output)
    xs = np.array([[i,i] for i in range(n_sample)])
    list_name = ['%d'%(i) for i in range(n_sample)]

    #* Built-in functions
    if True:

        mpRun = MultiProcessEvaluation(dim_input, dim_output, func=usr_func, n_process=n_process)

        _, ys = mpRun.evaluate(xs, list_name)

        for i in range(min(4, n_sample)):
            print('%3d x= '%(i), xs[i,:], ' y= ', ys[i,:])

    #* External functions
    if True:
        
        mpRun = MultiProcessEvaluation(dim_input, dim_output, n_process=n_process)

        _, ys = mpRun.evaluate(xs, list_name, prob=prob)

        for i in range(min(4, n_sample)):
            print('%3d x= '%(i), xs[i,:], ' y= ', ys[i,:])
