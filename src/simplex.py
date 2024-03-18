import numpy as np

class Simplex:
    def __init__(self, A, b, c,B=None,N=None,list_b=None,list_nb= None):

        self.A = A
        self.m, self.n = A.shape
        self.b = b
        self.c = c

        #apart
        self.list_b = list_b
        self.list_nb = list_nb
        
    def solve(self):
        print('Inici simplex primal amb regla de Bland ')

        print('Fase I')
        self.B , self.N, self.list_b, self.list_nb, iteracio= self.fase_1()

        #la fase1 retorna: B ,N,list_b,list_nb, iteracio
        #la fase2 retorna: iteracio, xb, z, r

        print(f'    Solució bàsica factible trobada, iteració {iteracio} ')

        print('Fase II')
        resposta = self.fase_2(it=iteracio)
        print(f'    Solució òptima trobada, iteració {resposta[0]}, z = {resposta[2]}')
        print('Fi simplex primal')

        print('')
        print('')

        print('solució òptima: ')
        print(f'vb = {self.B}')
        print(f'xb = {resposta[1]}')
        print(f'z = {resposta[2]}')
        print(f'r = {resposta[3]}')


        
    
    def fase_1(self):

        A_artificial = np.hstack((self.A, np.eye(self.m)))
        c_artificial = np.hstack((np.zeros(self.n), np.ones(self.m)))
        
        list_nb_artificial = np.array([list(range(self.n))])
        list_b_artificial = np.array([list(range(self.n, self.n + self.m))])

        problema_artificial = Simplex(A_artificial,self.b,c_artificial,list_b_artificial,list_nb_artificial)
        res = problema_artificial.fase_2()

        return problema_artificial.list_b,problema_artificial.list_nb,res[0]
        
    
    def fase_2(self,it=0):

        Ab = self.A[:,self.list_b]
        inversa = np.linalg.inv(Ab)
        xb = np.dot(inversa,self.b)
        if np.all(xb < 0):
            #existe elemento de xb negativo, por tanto infactible
            #0 es el indicador de infactibilidad
            return 0
        


        while True:
            it += 1
            #como Xn es un vector de 0, el coste del problema es solo el coste de las basicas
            cb = self.c[:,self.list_b]
            cn = self.c[:,self.list_nb]
            An = self.A[:,self.list_nb]
            z = np.dot(cb,xb)


            #calculamos los costes reducidos de las variables no basicas
            r = cn - cb @ inversa @ An

            if np.all(r>=0):
                #hemos encontrado el óptimo
                return it, xb,z,r
            
            #tota linea a continuacio només s'executa si no s'ha trobat optim

            valor_minim = np.min(r)
            # Encontramos los índices donde se encuentra el valor negativo mínimo
            possibles_q = np.where(r == valor_minim)
            q = np.min(possibles_q)
            
            #calculamos la direccion basica

            Aq = self.A[:q]
            db = -inversa @ Aq

            if np.all(db>=0):
                #el problema seria no acotado
                return 1
            
            #calculamos la theta

            db_minim = np.min(db)

            # Encontramos los índices donde se encuentra el valor negativo mínimo
            possibles_i = np.where(db == db_minim)
            indice_i = np.min(possibles_i)

            xi = xb[:indice_i]
            dbi = db[:indice_i]
            p = self.list_b[:indice_i]
            theta = -xi/dbi



            #ACTUALITZACIONS

            #actualització de Xb   z (comentar amb el flores)
            xb = xb + theta*db
            if np.all(xb < 0):
                #existe elemento de xb negativo, por tanto infactible
                #0 es el indicador de infactibilidad
                return 0
            
            #actualització de la inversa
            matriu_E = np.eye(self.m)
            columna_P = np.zeros((self.m, 1))

            for i in range(self.m):
                if i == p:
                    columna_P[i,0] = -1/dbi
                else:
                    columna_P[i,0] = -db[:i]/dbi

            matriu_E[:,p] = columna_P[:,0]

            inversa = matriu_E @ inversa

            #actualització de list_b,list_nb

            i_p = np.where(self.list_b == p)
            self.list_b[:i_p] = q

            i_q = np.where(self.list_nb == q)
            self.list_nb[:i_q] = p




    
    def solve2(self):
        while True:
            inv_base = np.linalg.inv(self.A[:, self.base])
            cb = self.c[self.base]
            costos_reducidos = self.c - cb.dot(inv_base.dot(self.A))
            
            # Regla de Bland para la variable entrante: elegir la de menor índice con costo reducido positivo
            q_candidates = [j for j, costo in enumerate(costos_reducidos) if costo > 0]
            if not q_candidates:
                # Solución óptima alcanzada
                valores_variables_basicas = inv_base.dot(self.b)
                valor_optimo = cb.dot(valores_variables_basicas)
                print("Solución óptima encontrada")
                print("Variables básicas:", self.base)
                print("Valor óptimo:", valor_optimo)
                return valores_variables_basicas, valor_optimo
            
            q = min(q_candidates)

            # Dirección de búsqueda y regla del mínimo cociente, incluyendo la regla de Bland
            direccion = inv_base.dot(self.A[:, q])
            ratios = [self.b[i] / direccion[i] if direccion[i] > 0 else np.inf for i in range(self.m)]
            p_candidates = [i for i, ratio in enumerate(ratios) if ratio != np.inf]
            if not p_candidates:
                print("El problema es no acotado.")
                return None
            
            p = p_candidates[np.argmin([ratios[i] for i in p_candidates])]

            if self.base[p] == q:
                # No hay cambio efectivo en la base, posiblemente estancado
                print("Potencial estancamiento detectado. Revisar implementación o criterios de parada.")
                return None
            
            self.base[p] = q  # Actualización de la base

            print(f"Iteración: Variables básicas {self.base}, Valor de función objetivo: {cb.dot(inv_base.dot(self.b))}")

if __name__  == "__main__":
    #funció de lectura per inicialitzar els paràmetres necessaris (A,b,c)
    A = np.array([[37, -3, 81, 65, -35, 85, -79, -59, -12, -29, 50, -1, -42, 53, 0, 0, 0, 0, 0, 0],
                [-64, 26, 80, -25, 22, 88, -10, -19, 36, -8, -32, -26, -70, 79, 0, 0, 0, 0, 0, 0],
                [-97, 62, -67, 36, 63, 41, 32, 16, 77, -17, -93, -26, -79, 67, 0, 0, 0, 0, 0, 0],
                [-31, 90, 10, 29, -59, 75, 69, 72, 59, -62, -44, -63, -95, -38, 0, 0, 0, 0, 0, 0],
                [64, 82, 91, 71, 88, 54, 97, 95, 95, 83, 71, 63, 82, 51, 1, 0, 0, 0, 0, 0],
                [35, 19, 74, -81, 8, 88, 68, 30, 34, 42, 0, -70, 41, -14, 0, 1, 0, 0, 0, 0],
                [90, -65, 79, 59, -67, 24, 49, 49, 32, 16, 28, -53, 39, -21, 0, 0, 1, 0, 0, 0],
                [3, 81, 55, -54, 73, -31, -31, 71, -25, 49, 63, 49, -42, -37, 0, 0, 0, 1, 0, 0],
                [60, 88, -89, -29, -90, 99, 35, 26, -57, -58, 53, 33, -78, 29, 0, 0, 0, 0, 1, 0],
                [-46, 11, 28, 46, 32, 39, 82, -15, 42, 37, 97, -53, 48, 22, 0, 0, 0, 0, 0, 1]])

    b = np.array([111, 77, 15, 12, 1088, 275, 260, 225, 23, 371])
    c = np.array([-55, 8, 38, 33, 0, -63, 54, 58, -81, 3, -59, -14, -10, 48, 0, 0, 0, 0, 0 ,0])

    problem = Simplex(A,b,c)
    problem.solve()