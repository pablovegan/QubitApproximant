# coding=UTF-8

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from typing import Optional
import pickle

import qml_model
import test_functions
import qml_optimizer

def error_perceptron(φ: np.ndarray,
                    function: str = 'gaussian',
                    f_params: dict = {'mean': 0.0, 'std': 0.5, 'coef': None},
                    interval: tuple = (-1,1),
                    method: str = 'simpson',
                    probability: bool = True):
    """"
    Error in the approximation of the function by the qubit perceptron.
    Returns the error measured in different norms.
    
    """
    layers = φ.size // 4
    w, θ = φ[0:layers], φ[layers:].reshape(3, layers)

    # Seleccionamos la función a aproximar
    fun = getattr(test_functions.Test_Functions, function)
    # Norma L2
    diff_l2 = lambda x: (fun(x, **f_params) - qml_model.evaluate_model(x, θ, w, probability).real)**2
    f2_theo = lambda x: fun(x, **f_params)**2
    # Norma L1
    diff_abs = lambda x: np.abs(fun(x, **f_params) - qml_model.evaluate_model(x, θ, w, probability).real)
    f_theo_abs = lambda x: np.abs(fun(x, **f_params))
    # Fidelity (no hace falta conjugar f porque el modelo es real)
    prod_re = lambda x: np.real(fun(x, **f_params)*qml_model.evaluate_model(x, θ, w, probability))
    prod_im = lambda x: np.imag(fun(x, **f_params)*qml_model.evaluate_model(x, θ, w, probability))
    f2_approx = lambda x: np.abs(qml_model.evaluate_model(x, θ, w, probability))**2
    # Norma infinito
    y = np.linspace(interval[0], interval[1], 10000)
    error_inf = np.max(np.abs(fun(y, **f_params) - qml_model.evaluate_model(y, θ, w, probability).real))
    # Seleccionamos el método de integración
    if method == 'simpson':
        # L2 calculation
        error_l2 = np.sqrt(integrate.simpson(diff_l2(y), y))/np.sqrt(integrate.simpson(f2_theo(y), y))
        # L2 calculation
        error_l1 = integrate.simpson(diff_abs(y), y)/integrate.simpson(f_theo_abs(y), y)
        # Fidelity calculation
        int_prod_squared = integrate.simpson(prod_re(y), y)**2+integrate.simpson(prod_im(y), y)**2
        error_infid = 1-int_prod_squared/(integrate.simpson(f2_approx(y), y) * integrate.simpson(f2_theo(y), y))
        return error_l2, error_l1, error_inf, error_infid

    if method == 'quad':
        error_l2 = np.sqrt(integrate.quad(diff_l2, interval[0], interval[1], limit=100))
        error_l1 = integrate.quad(diff_abs, interval[0], interval[1], limit=100)
        int_prod_squared = integrate.simpson(prod_re, interval[0], interval[1], limit=100)[0]**2+integrate.simpson(prod_im, interval[0], interval[1], limit=100)[0]**2
        error_infid = 1 - int_prod_squared/(integrate.simpson(f2_approx, interval[0], interval[1], limit=100)[0] * integrate.simpson(f2_theo, interval[0], interval[1], limit=100))[0]
        return error_l2[0], error_l1[0], error_inf, error_infid


def plot_errores(layer_list, l2_list, l1_list, inf_list, infid_list, cost_error, function):
    plt.close('all')
    fig, ax = plt.subplots(1,1)
    ax.plot(layer_list, l2_list, linestyle='-', marker='o', markersize = 6, color='#1f77b4', label='L2 norm')
    ax.plot(layer_list, l1_list, linestyle='-', marker='^', markersize = 6, color='#ff7f0e', label='L1 norm')
    ax.plot(layer_list, inf_list, linestyle='-', marker='D',markersize = 6, color='#2ca02c', label='Infinity norm')
    ax.plot(layer_list, infid_list, linestyle='-', marker='*',markersize = 6, color='crimson', label='Infidelity')
    ax.plot(layer_list, cost_error, linestyle='-', marker='*',markersize = 6, color='olive', label='Coste')
    

    ax.set_title('Error vs Number of layers (' + function + ')')
    ax.set_xlabel('Layers')
    ax.set_ylabel('Error')
    ax.legend(loc='upper right', fontsize='large')
    plt.yscale('log', subs = [2, 3, 4, 5, 6, 7, 8, 9])
    plt.show()


def graficas_errores(min_layers, max_layers,
                    x: Optional[np.ndarray] = None,
                    f: Optional[np.ndarray] = None,
                    grid_size: int = 31,
                    function: str = 'gaussian',
                    f_params: dict = {'mean': 0.0, 'std': 2, 'coef': 1},
                    interval: tuple = (-1,1),
                    int_method: str = 'simpson',
                    opt_method: str = 'L-BFGS-B',
                    seed: int = 4,
                    φ_init: Optional[np.ndarray] = None,
                    show_plot: bool = False,
                    show_final_plot: bool = True,
                    show_error_plot: bool = True,
                    show_diff = False,
                    print_cost: bool = False,
                    cost_fun: str = 'sqrt',
                    incremental_opt: bool = True,
                    print_params: bool = True,
                    cc: float = 0.3,
                    new_layer_position: str = 'random',
                    new_layer_coef: float = 0.2,
                    plot_cost_error: bool = False,
                    method_params: dict = {'n_iter': 800, 'alpha': 0.01, 'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-8}):
    
    Test = test_functions.Test_Functions()
    l2_list, l1_list, inf_list, infid_list = [], [], [], []
    layer_list = list(range(min_layers, max_layers+1))

    x, f = Test.plot(grid_size, function, f_params, interval = interval, show_plot = show_plot)
    probability = (f >= 0).all()
    
    np.random.seed(seed)
    if φ_init is None:
        # cc deberá ser un valor pequeño en principio
        φ = cc * np.random.randn(min_layers + 3*min_layers)
        if print_params: print('Parámetros iniciales: ', φ)
    else: φ = φ_init

    max_diff = np.zeros(max_layers-min_layers+1)
    cost_error = np.zeros(max_layers-min_layers+1)
    for i, layer in enumerate(layer_list):
        φ, result = qml_optimizer.train_perceptron(x, f, layers = layer, probability = probability, opt_method = opt_method , seed = seed,
                                 φ_init = φ, show_plot = show_plot, method_params = method_params, print_cost = print_cost,
                                 plot_title = function + ' optimized with ' + opt_method, cost_fun = cost_fun)
        # print('Los parámetros óptimos en la capa {layer} son {φ}.\n'.format(layer = layer, φ=φ))
        error_l2, error_l1, error_inf, error_infid = error_perceptron(φ, function, f_params, interval, int_method, probability)
        
        ω, θ = qml_optimizer.split(φ)
        cost_error[i] = qml_model.cost(x,f,θ,ω,probability)
        # Guardamos la diferencia entre el φ optimizado de la anterior capa y de esta
        if show_diff and (layer > min_layers):
            if new_layer_position == 'final':
                diff_φ = φ.reshape(4,layer).T.flatten()[0:4*(layer-1)] - φ_old
            elif new_layer_position == 'initial':
                diff_φ = φ.reshape(4,layer).T.flatten()[4:4*layer] - φ_old
            max_diff[i] = np.abs(diff_φ).max()
            print('La diferencia entre los parámetros optimizados es {diff} y su máximo es {max_diff}.\n'.format(diff = diff_φ, max_diff = max_diff[i]))
        φ_old = φ

        l2_list.append(error_l2)
        l1_list.append(error_l1)
        inf_list.append(error_inf)
        infid_list.append(error_infid)

        if layer == max_layers:
            if show_final_plot:
                ω, θ = φ[0:layer], φ[layer:].reshape(3, layer) 
                f_approx = qml_model.evaluate_model(x, θ, ω, probability)
                plt.close('all')
                plt.plot(x, f)
                plt.plot(x, f_approx.real)
                plt.title(function + ' optimization with ' + opt_method)
                plt.show()
            break

        if incremental_opt is True:
            # Inicializamos una nueva capa en la posición indicada
            if new_layer_position == 'random':
                i = np.random.randint(0, high=layer+1, dtype=int)
            elif new_layer_position == 'final':
                i = layer
            elif new_layer_position == 'initial':
                i = 0
            elif new_layer_position == 'middle':
                i = min_layers + (layer-min_layers)//2 
            else: raise ValueError('El valor de new_layer_position = {a} no es válido.'.format(a = new_layer_position))
            # Añadimos la nueva capa con valores cercanos a 0
            new_layer_val = new_layer_coef * np.random.randn(4)
            #new_layer_val = 0.3/(i+1) * np.random.randn(4)
            φ = np.insert(φ, i, new_layer_val[0])  # phi [w1, ...wn, theta1, theta2, theta3]
            φ = np.insert(φ, i+1+layer, new_layer_val[1])
            φ = np.insert(φ, i+2+2*layer, new_layer_val[2])
            φ = np.insert(φ, i+3+3*layer, new_layer_val[3])

        else:
            φ = cc * np.random.randn(layer+1 + 3*layer+3)
        # print('Los parámetros con capa añadida son {φ}.\n'.format(φ=φ))

    if show_diff:
        plt.plot(layer_list[1:], max_diff[1:],marker='o',markersize = 6, color='crimson')
        plt.show()
    if print_params: print('Parámetros finales: ', φ)
    # Hacemos una integración numérica para calcular el error de la aproximación
    if show_error_plot:
        plot_errores(layer_list, l2_list, l1_list, inf_list, infid_list, cost_error, function)

    if plot_cost_error:
        plt.close()
        plt.figure(figsize=(6, 5), dpi=80)
        plt.plot(layer_list, cost_error, ls = '--', marker = '^', ms = 14)
        plt.yscale('log', subs = [2, 3, 4, 5, 6, 7, 8, 9])
        plt.show()

    return layer_list, l2_list, l1_list, inf_list, infid_list, cost_error


def mean_seed_errores(min_layers, max_layers,
                    x: Optional[np.ndarray] = None,
                    f: Optional[np.ndarray] = None,
                    grid_size: int = 31,
                    function: str = 'gaussian',
                    f_params: dict = {'mean': 0.0, 'std': 2, 'coef': 1},
                    interval: tuple = (-1,1),
                    int_method: str = 'simpson',
                    opt_method: str = 'L-BFGS-B',
                    φ_init: Optional[np.ndarray] = None,
                    show_plot: bool = False,
                    show_final_plot: bool = True,
                    show_error_plot: bool = True,
                    show_box_plot = True,
                    show_diff = False,
                    print_cost: bool = False,
                    cost_fun: str = 'sqrt',
                    incremental_opt: bool = True,
                    print_params: bool = True,
                    cc: float = 0.3,
                    new_layer_position: str = 'random',
                    new_layer_coef: float = 10e-4,
                    plot_cost_error: bool = False,
                    num_seed = 15,
                    filename = '',
                    method_params: dict = {'n_iter': 800, 'alpha': 0.01, 'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-8}):          

    num_layer = max_layers - min_layers + 1
    l2, l1, inf, fid, cost = np.zeros(num_layer), np.zeros(num_layer), np.zeros(num_layer), np.zeros(num_layer), np.zeros(num_layer)

    cost_array = np.zeros((num_seed,num_layer))
    l1_array = np.zeros((num_seed,num_layer))
    l2_array = np.zeros((num_seed,num_layer))
    fid_array = np.zeros((num_seed,num_layer))
    inf_array = np.zeros((num_seed,num_layer))

    for i, seed in enumerate(np.random.choice(range(0,100), num_seed, replace=False)):
        layer_list, l2_list, l1_list, inf_list, fid_list, cost_list = graficas_errores(min_layers = min_layers, max_layers = max_layers, x = x, f = f, grid_size = grid_size, function = function, 
            f_params = f_params,interval = interval,int_method = int_method,opt_method = opt_method,φ_init = φ_init,
            show_plot = show_plot, show_final_plot = show_final_plot,show_error_plot = show_error_plot,show_diff = show_diff,
            print_cost = print_cost,cost_fun = cost_fun,incremental_opt = incremental_opt,print_params = print_params, cc = cc,
            new_layer_position = new_layer_position,new_layer_coef = new_layer_coef,plot_cost_error = plot_cost_error,
            method_params=method_params, seed = seed)

        # Seeds en el eje 0 y capas en el eje 1. Queremos las seeds en cada box plot.'''
        cost_array[i,:]= np.array(cost_list)
        l1_array[i,:]= np.array(l1_list)
        l2_array[i,:]= np.array(l2_list)
        fid_array[i,:]= np.array(fid_list)
        inf_array[i,:]= np.array(inf_list)

    with open(filename+'.pkl', 'wb') as file:
        pickle.dump((layer_list, l2_array, l1_array, inf_array, fid_array, cost_array), file)

    if show_box_plot:
        box_plot_errores(layer_list, cost_array, function)
    return

    l2 = l2/num_seed
    l1 = l1/num_seed
    inf = inf/num_seed
    fid = fid/num_seed
    cost = cost/num_seed

    plot_errores(layer_list, l2, l1, inf, fid, cost, function)
    return layer_list, l2, l1, inf, fid, cost
    

def box_plot_errores(layer_list, cost_array, function):
    # plot
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    ax[0].boxplot(cost_array, meanline = True, patch_artist = True, medianprops={"color": "crimson", "linewidth": 1},
                boxprops={"facecolor": "lightsteelblue", "edgecolor": "black", "linewidth": 0.5})
    ''', positions=layer_list, widths=1.5, patch_artist=True,
                showmeans=False, showfliers=False,
                medianprops={"color": "white", "linewidth": 0.5},
                boxprops={"facecolor": "lightsteelblue", "edgecolor": "white", "linewidth": 0.5},
                whiskerprops={"color": "C0", "linewidth": 1.5})'''
    ax[0].set_title(function)
    ax[0].set_xlabel('Capas')
    ax[0].set_ylabel('Error')
    ax[0].set_yscale('log')

    violin = ax[1].violinplot(cost_array, showmeans=True, showmedians=True)
    ax[1].set_title(function)
    ax[1].set_xlabel('Capas')
    ax[1].set_xticks([y + 1 for y in range(cost_array.shape[1])])
    ax[1].set_ylabel('Error')
    ax[1].set_yscale('log')
    
    violin['cmedians'].set_color('crimson')

    plt.show()