import scipy.stats as stats
from numpy.random import binomial
import numpy as np
import math
from numpy import loadtxt 

def intervalle_confiance(e, alp):
    """Calcule un intervalle de confiance de niveau 1-alpha pour 
    le paramètre p d'un échantillon de Bernoulli. 
    

    Parameters
    ----------
    e : ndarray 
        Réalisation d'un échantillon de Bernoulli. 
    alp : scalar (dans (0, 1)) 
        Paramètre qui détermine le niveau de confiance. 

    Returns
    -------
    borne_inferieure : real
        Borne inferieure de l'intervalle de confiance de niveau 1-alpha. 

    borne_superieure : real
        Borne superieure de l'intervalle de confiance de niveau 1-alpha. 
    """ 


    n = len(e)
    p_chap = np.mean(e)
    
    t = stats.norm.ppf(1 - alp / 2)
    a = t**2 / n
    delta = 4 * p_chap * (1 - p_chap) * a + a**2

    # Calcul des bornes de l'intervalle de confiance
    p_chap_inf = (p_chap + a / 2 - math.sqrt(delta) / 2) / (1 + a)
    p_chap_sup = (p_chap + a / 2 + math.sqrt(delta) / 2) / (1 + a)
    
    return (p_chap_inf, p_chap_sup)

# Exemple d'utilisation
"""
p = 0.3
n = 1000
alpha = 0.05

# Simulation d'un échantillon de loi de Bernouilli de paramètre p.
echantillon = binomial(1, p, n)


intervalle_confiance = intervalle_confiance(echantillon, alpha)
print(f"Intervalle de confiance à {(1-alpha)*100}% : {intervalle_confiance}")
"""
# Plusieurs échantillons de bernoulli
p = 0.3
taille_ech = 1000
n = 1000
alpha = 0.05



# Générer plusieurs échantillons de loi de Bernoulli
M = [binomial(1, p, taille_ech) for _ in range(n)]
# M = nombre des echantiollons
# Calculer la proportion de réalisations pour lesquelles p est dans l'intervalle de confiance
proportion = 0
# T intervalle de confiance actuel
for i in M:
    T = intervalle_confiance(i, alpha)
    if T[0] <= p <= T[1]:
        proportion += 1

proportion /= taille_ech

print(f"Proportion de réalisations pour lesquelles p est dans l'intervalle de confiance : {(proportion) * 100}%")

#precision
precisions = []

for i in M:
    T = intervalle_confiance(i, alpha)
    precision = (T[0] <= p <= T[1])
    precisions.append(precision)

moyenne_empirique_precisions = np.mean(precisions)

print(f"Moyenne empirique des précisions : {moyenne_empirique_precisions * 100}%")


def nombre_dobservations_precision (precision,alp):
    t = stats.norm.ppf(1 - alp / 2)
    n = (t / precision)**2
    return int(n) + 1  # Arrondir à l'entier supérieur, car le nombre d'observations doit être entier
# Exemple d'utilisation
precision_voulue = 0.05  # Par exemple, une précision de 5%
niveau_confiance = 0.05  # Un niveau de confiance de 95% correspond à alpha = 0.05

nombre_observations = nombre_dobservations_precision(precision_voulue, niveau_confiance)
print(f"Nombre d'observations suffisant : {nombre_observations}")

#Derniere question
#Valeurs 1
precision_voulue2 = 0.01  # Par exemple, une précision de 5%
niveau_confiance = 0.05  # Un niobject of type 'int' has no len()veau de confiance de 95% correspond à alpha = 0.05

nombre_obs = nombre_dobservations_precision(precision_voulue2, niveau_confiance)
print(f"Nombre d'observations suffisant : {nombre_obs}")

#Valeurs 2
precision_voulue3 = 0.005  # Par exemple, une précision de 5%
niveau_confiance = 0.05  # Un niveau de confiance de 95% correspond à alpha = 0.05

nombre_obs2 = nombre_dobservations_precision(precision_voulue3, niveau_confiance)
print(f"Nombre d'observations suffisant : {nombre_obs2}")

#Valeurs 3
precision_voulue3 = 0.01  # Par exemple, une précision de 5%
niveau_confiance = 0.025  # Un niveau de confiance de 95% correspond à alpha = 0.05

nombre_obs2 = nombre_dobservations_precision(precision_voulue3, niveau_confiance)
print(f"Nombre d'observations suffisant : {nombre_obs2}")

#Approximation de l'integrale de gausse tronquée
E = nombre_dobservations_precision (0.01, 0.05)
print(E)


#Estimation de temps d'extinction

frontales = loadtxt('frontales.txt')
#Estimons cette probabilité par intervalle de confiance à partir des données.
"""N = len(frontales)
U = intervalle_confiance(N, 0.05)
print (U)
"""
#Approximation de l'intégrale de Gauss tronquée Estimation du paramètre de taux lambda à partir des données
lambda_estimate = 1 / np.mean(frontales)

# Estimation de la probabilité que les lampes s'éteignent avant 3 heures
time_limit = 3
probability_estimate = 1 - np.exp(-lambda_estimate * time_limit)

# Calcul de l'intervalle de confiance
alpha = 0.05  # Niveau de confiance à 95%
n = len(frontales)

# Calcul de l'écart type de l'estimateur
std_error = np.sqrt(lambda_estimate / n)

# Calcul de l'intervalle de confiance
conf_interval = stats.norm.interval(1 - alpha, loc=probability_estimate, scale=std_error)

# Affichage des résultats
print(f"Estimation de la probabilité : {probability_estimate:.4f}")
print(f"Intervalle de confiance à {100*(1-alpha)}% : ({conf_interval[0]:.4f}, {conf_interval[1]:.4f})")
