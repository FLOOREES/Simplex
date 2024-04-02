#IMPORTS QUE ES FAN SERVIR AL MODEL DE SIMPLEX
import numpy as np
from scipy.optimize import linprog

#CODI DE NUMPY PERQUE LES IMPRESIONS PER PANTALLA NO SIGUIN EN NOTACIÓ CIENTÍFICA
np.set_printoptions(suppress=True)

class Simplex:
    def __init__(self, A, b, c,list_b=None):

        self.A = A
        self.m, self.n = A.shape
        self.b = b
        self.c = c

        #apart
        self.list_b = list_b
        
    def solve(self):
        print('')
        print('-------INICI SIMPLEX PRIMAL AMB LA REGLA DE BLAND---------------')
        print('')
        print('-------------------------FASE I-------------------------------')
        print('')
        fase1 = self._fase_1()
        if fase1 == 1:
            print('PROBLEMA NO ACOTAT: fase 1')
        elif fase1 == 0:
            print('PROBLEMA NO FACTIBLE: fase 1')
        else:

            self.list_b = fase1[0]
            iteracio = fase1[1]

            #la fase1 retorna: list_b, iteracio
            #la fase2 retorna: iteracio, xb, z, r

            print(f'    Solució bàsica factible trobada, iteració {iteracio} ')
            print('')
            print('')
            print('-------------------------FASE II-----------------------------')
            resposta = self._fase_2(it=iteracio)
            if resposta == 0: 
                print('PROBLEMA NO FACTIBLE')
            
            elif resposta == 1: 
                print('PROBLEMA NO ACOTAT')
            else:

                print(f'Solució òptima trobada, iteració {resposta[0]}, z = {resposta[2]}')

                print('------------------------FI SIMPLEX PRIMAL-------------------------------')

                print('')
                print('')

                #arrodonim respostes per tenir output bonic
                res_xb = np.round(resposta[1],4)
                res_r = np.round(resposta[3],4)

                print('---------------SOLUCIÓ ÒPTIMA:---------------------')
                print(f'vb = {self.list_b}')
                print(f'xb = {res_xb}')
                print(f'z = {resposta[2]}')
                print(f'r = {res_r}')


        
    
    def _fase_1(self):

        #-------------------------CREEM LES VARIABLES ARTIFICIALS-------------------------
        A_artificial = np.hstack((self.A, np.eye(self.m)))
        c_artificial = np.hstack((np.zeros(self.n), np.ones(self.m)))
        list_b_artificial = np.array(list(range(self.n, self.n + self.m)))


        #------------------------CREEM UN PROBLEMA SIMPLEX DEL PROBLEMA ARTIFICIAL-----------------
        problema_artificial = Simplex(A_artificial,self.b,c_artificial,list_b_artificial)
        res = problema_artificial._fase_2()
        if res == 1:
            return 1
        if res == 0 or res[2] > 0.001:
            return 0
        return problema_artificial.list_b,res[0]
        
    
    def _fase_2(self,it=0):

        #-------------INICIALITZEM ELS NOSTRES VALORS INICIALS ---------
        variables = np.array(list(range(self.n)))
        list_nb = np.setdiff1d(variables,self.list_b)
        Ab = self.A[:, self.list_b]
        inversa = np.linalg.inv(Ab)
        xb = np.dot(inversa,self.b)
        cb = self.c[self.list_b]        
        z = np.dot(cb,xb)
        z_round = np.round(z,4)

        print('***************************************')
        print('DATOS INICIALES:')
        print(f'VARIABLES BASICAS = {self.list_b}')
        print(f'VARIABLES NO BASICAS = {list_nb}')
        print(f'z = {z_round}')
        print('***************************************')
        print('')

        #---------------ITEREM EN EL SIMPLEX PRIMAL-------------------------------------------
        while True:
            it += 1


            #--------------------CALCUL DELS COSTOS REDUITS DE LES VARIABLES NO BASIQUES------------------
            cn = self.c[list_nb]
            An = self.A[:,list_nb]
            cb = self.c[self.list_b]        
            r = cn - np.dot( np.dot(cb,inversa),An)

            #--------------------------------MIRAR SI ES ÒPTIM-----------------------------------------
            if np.all(r>=0):
                #HEM TROBAT EL VALOR ÒPTIM
                return it, xb,z_round,r
            
            #NO HEM TROBAT EL VALOR ÒPTIM: 
            
            #------------------------TREIEM 'q' QUE ENTRA COM A BÀSICA---------------------------

            valor_minim = float('inf')
            q = None
            for index, valor in enumerate(r):
                if valor < valor_minim:
                    valor_minim = valor
                    q = list_nb[index]
                elif valor == valor_minim:
                    #REGLA DE BLAND
                    if list_nb[index] < q:
                        q = list_nb[index]



            
            #------------------------CALCUL DE LA DIRECCIO BASICA---------------------------

            Aq = self.A[:,q]

            db = np.dot(-1*inversa,Aq)
            if np.all(db>=0):
                #el problema seria no acotado
                return 1
            
            #----------------CALCUL DE LA THETA Y 'p' QUE SURT COM A NO BÀSICA-----------------------------

            theta = float('inf')
            p = None

            for i,dbi in enumerate(db):
                if dbi<0:
                    if (-xb[i]/dbi)<theta:
                        theta = (-xb[i]/dbi)
                        p = i

            #------------------ACTUALITZACIONS-----------------------
                        
            #actualització z 
            i_q = np.where(list_nb == q)
            i_q = np.squeeze(i_q)
            z = z + theta * r[i_q]

            #actualització de Xb
            xb = xb + theta*db
            xq = theta
            xb[p] = xq 

            if np.all(xb < 0):
                #existe elemento de xb negativo, por tanto infactible
                #0 es el indicador de infactibilidad
                return 0

            #actualització de la inversa
            matriu_E = np.eye(self.m)
            columna_P = np.zeros((self.m, 1))
            for i in range(self.m):
                if i == p:
                    columna_P[i,0] = -1/db[p]
                else:
                    columna_P[i,0] = -db[i]/db[p]
            matriu_E[:,p] = columna_P[:,0]
            inversa = np.dot(matriu_E,inversa)

            #actualització de list_b,list_nb
            valor_p = self.list_b[p]
            self.list_b[p] = q
            list_nb[i_q] = valor_p

            #-------------------------PRINTS DE CADA ITERACIÓ----------------------------------------
            theta_round = np.round(theta,4)
            z_round = np.round(z,4)

            print(f'iteració  {it}: Var ENTRA = {q}, VAR SURT= {valor_p}, theta*= {theta_round}, z={z_round}')
            print(f'VARIABLES BASICAS = {self.list_b}')
            print(f'VARIABLES NO BASICAS = {list_nb}')
            print('')




#------------------------------------EXEMPLE DE SIMPLEX---------------------------------------------------

if __name__  == "__main__":

    #HEM DE CREAR UNA funció de lectura per inicialitzar els paràmetres necessaris (A,b,c)
    with open('Practica_simplex/problemes.txt', 'r') as file:
        lineas = file.readlines()
    k=0
    i = 0
    while i < len(lineas):
        if lineas[i].strip().startswith('c='):
            k+=1
            print(f"Resultat del problema {k}: ")
            c = np.fromstring(lineas[i+1].strip(), sep=' ')
            i += 3
        elif lineas[i].strip().startswith('A='):
            A = []
            i += 1  # Avanza a la primera línea de valores de 'A'
            while not lineas[i].strip().startswith('b='):
                if lineas[i].strip():  # Si la línea no está en blanco
                    fila = np.fromstring(lineas[i].strip(), sep=' ')
                    A.append(fila)
                i += 1
            A = np.vstack(A)
        elif lineas[i].strip().startswith('b='):
            b = np.fromstring(lineas[i+1].strip(), sep=' ')
            i += 3
            simplex = Simplex(A,b,c)
            simplex.solve()
            res = linprog(c, A_eq=A, b_eq=b, method='highs')
            print(f'\nResultat de la funció linprog de scipy: z = {res.fun}')
            print('-'*150)
            print('')