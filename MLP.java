

import java.io.*;
/**
  * @author hubert.cardot
  * modifier par Yongzhi
 */
public class MLP {  // pg du MLP, rÈseau de neurones ‡ rÈtropropagation

    static int NbClasses=3, NbCaract=4, NbEx=50, NbExApprent=25;
    static int NbCouches=3, NbCaches=6, NbApprent=2000; 
    static int NbNeurones[]={NbCaract+1, NbCaches+1, NbClasses}; //+1 pour neurone fixe我们对神经网络的结构图的讨论中都没有提到偏置节点（bias unit）。事实上，这些节点是默认存在的。它本质上是一个只含有存储功能，且存储值永远为1的单元。在神经网络的每个层次中，除了输出层以外，都会含有这样一个偏置单元
    static Double data[][][] = new Double[NbClasses][NbEx][NbCaract];
    static Double poids[][][], N[][], S[][], coeffApprent=0.01, coeffSigmoide=2.0/3;//coefApprent 学习速率
    
    private static Double fSigmoide(Double x)  {       // f()
    	return Math.tanh(coeffSigmoide*x); } 
    private static Double dfSigmoide(Double x) {       // df()
    	return coeffSigmoide/Math.pow(Math.cosh(coeffSigmoide*x),2); } 
    
    public static void main(String[] args) {
        System.out.println("Caches="+NbCaches+" App="+NbApprent+" coef="+coeffApprent);
        initialisation();
        apprentissage();
        evaluation();
    }   

    private static void initialisation() {
        lectureFichier(); 
        //Allocation et initialisation alÈatoire des poids 随机生成权重
        poids= new Double[NbCouches-1][][];
        // poids[2] 3 couche, 2 ensemble de poids
        for (int couche=0; couche<NbCouches-1; couche++) {
        	poids[couche] = new Double[NbNeurones[couche+1]][];
        	// poids[0] = [NbCaches+1][]
        	// poids[1] = [NbClasse][]
        	for (int i=0; i<NbNeurones[couche+1]; i++) {
        		poids[couche][i] = new Double[NbNeurones[couche]];
        		//poids[0][couche droite][couche gauche] chaque poid ‡ gauche
        		for (int j=0; j<NbNeurones[couche]; j++) {
        			poids[couche][i][j] = (Math.random()-0.5)/10; //dans [-0,05; +0,05[
        		}
        	}
        }
        //Allocation des Ètats internes N et des sorties S des neurones
        N = new Double[NbCouches][];//NbCouches = 3
        S = new Double[NbCouches][];
        for (int couche=0; couche<NbCouches; couche++) {
        	N[couche] = new Double[NbNeurones[couche]];
        	S[couche] = new Double[NbNeurones[couche]];
        }
    }
 
    // A faire
    private static void apprentissage() {  
    	
    	for(int n = 0; n < NbApprent; n++)//NbApprent =2000
    	{
    		for(int i = 0; i < NbClasses; i++)
    		{
    			int j = (int) (Math.random() * NbExApprent);//随机选取若干个Ex作为Apprentissage
        		propagation(data[i][j]);
        		retropropagation(i);
    		}
    	}
    } 
    private static int indexMax(Double S[])
    {
    	int key = 0;
    	for(int i = 0; i < S.length; i++)
    	{
    		if( S[i] > S[key])
    		{
    			key = i;
    		}
    	}
    	return key;
    }
    
    private static void evaluation() {
        int classeTrouvee, Ok=0, PasOk=0;
        for(int i=0; i<NbClasses; i++) {
            for(int j=NbExApprent; j<NbEx; j++) { // parcourt les ex. de test
                //---------- ‡ faire              // calcul des N et S des neurones
            	propagation(data[i][j]);
                classeTrouvee = indexMax(S[NbCouches-1]);
                // recherche max parmi les sorties RN
                
                System.out.println(j+". classe" +i+" classe trouvÈe "+classeTrouvee);
                if (i==classeTrouvee) Ok++; else PasOk++;
            }
        }
        System.out.println("Taux de reconnaissance : "+(Ok*100./(Ok+PasOk)));
    }
    
    private static void propagation(Double X[]) {
    	//Couche d'entrÈe 输入层
    	for(int i = 0; i < NbNeurones[0] - 1; i++)
    	{
    		S[0][i] = X[i]; //S[0][i]是输入层 将Ex作为输入层
    		//System.out.println("0"+","+i+" S"+S[0][i]);
    	} 
    	
    	//Autre couches 其它层
    	for(int couche = 1; couche < NbCouches; couche++)
    	{
    		S[couche-1][NbNeurones[couche - 1]-1] = +1.0; //neurone fixe ‡ +1 S[0][4] = 1固定多出来的那个为1
    		for(int i = 0; i < NbNeurones[couche];i++)
    		{
    			Double somme = 0.0;
    			for(int j = 0; j < NbNeurones[couche-1];j++)
    			{
    				somme += S[couche -1][j]*poids[couche-1][i][j];  				
    			}
    			N[couche][i] = somme;
    			S[couche][i] = fSigmoide(N[couche][i]);
    			//System.out.println(i + " " + j + " " + N[i][j] + " " + S[i][j] + " ");
    			//System.out.println(couche+","+i+" S:"+S[couche][i]);
    			//System.out.println(couche+","+i+" N"+N[couche][i]);
    		}
    	}
    }
/*
 * 在神经网络模型中，由于结构复杂，每次计算梯度的代价很大。因此还需要使用反向传播算法。反向传播算法是利用了神经网络的结构进行的计算。不一次计算所有参数的梯度，而是从后往前。首先计算输出层的梯度，然后是第二个参数矩阵的梯度，接着是中间层的梯度，再然后是第一个参数矩阵的梯度，最后是输入层的梯度。计算结束以后，所要的两个参数矩阵的梯度就都有了。

　　反向传播算法可以直观的理解为下图。梯度的计算从后往前，一层层反向传播。前缀E代表着相对导数的意思。
 */
    
    
    private static void retropropagation(int classe) {
    	//Allocation tableau delta (delta du cours) 误差
    	Double delta[][] = new Double[NbCouches - 1][];
    	for(int couche = 0; couche < NbCouches - 1; couche++)
    	{
    		delta[couche] = new Double[NbNeurones[couche+1]];
    	}
    	
    	//Calcul de l'delta
    	for( int couche = NbCouches - 1; couche > 0; couche--)
    	{
    		if(couche == NbCouches - 1)
    		{//couche sortie
    			Double desiree;
    			for(int i = 0; i < NbNeurones[couche]; i++)
    			{
    				if( i == classe ) 
    					desiree = +1.0;
    				else 
						desiree = -1.0;
					
    				delta[couche-1][i] = (S[couche][i]-desiree)*dfSigmoide(N[couche][i]);
    			}
    		}
    		else {//Autre couche
				for(int j = 0; j < NbNeurones[couche]-1;j++)
				{
					delta[couche-1][j] = 0.0;
					for(int i = 0; i < NbNeurones[couche+1]; i++)
					{
						delta[couche-1][j] += (delta[couche][i] * poids[couche][i][j] * dfSigmoide(N[couche][j]));
					}
				}
			}
    	}
    	
    	//Mise ‡ jour des poids
    	for( int couche = 0; couche < NbCouches - 1; couche++)
    	{
    		for( int j = 0; j < NbNeurones[couche]; j++)
    		{
    			for( int i = 0; i < NbNeurones[couche+1]; i++)
    			{
    				//on n'apprent pas vers les neurones fixes
    				if(couche == NbCouches -2 || i != NbNeurones[couche+1]-1 )
    					poids[couche][i][j] += -coeffApprent *delta[couche][i] * S[couche][j];
    			}
    		}
    	}
    	
    }   
    
    private static void lectureFichier() {
        // lecture des donnÈes ‡† partir du fichier iris.data
        String ligne, sousChaine;
        int classe=0, n=0;
        try {
             BufferedReader fic=new BufferedReader(new FileReader("iris.data"));
             while ((ligne=fic.readLine())!=null) {
                for(int i=0;i<NbCaract;i++) {
                    sousChaine = ligne.substring(i*NbCaract, i*NbCaract+3);
                    data[classe][n][i] = Double.parseDouble(sousChaine);
                    //System.out.println(data[classe][n][i]+" "+classe+" "+n);
                }
                if (++n==NbEx) { n=0; classe++; }
             }
        }
        catch (Exception e) { System.out.println(e.toString()); }
    }
}  //------------------fin classe MLP2--------------------
