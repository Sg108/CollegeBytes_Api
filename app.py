from flask import Flask,request,jsonify
import pandas as pd
import numpy as np
app = Flask(__name__)


def find_correlation_between_two_users(f: pd.DataFrame, u1: int,newUser:pd.Series):
    q=f.iloc[u1]
    
    
    l=[]
    for r,e in zip(q,newUser):
      l.append([r,e])
    arr = np.array(l) 

    a = pd.DataFrame(arr, columns = ["1","2"])
    rated_movies_by_both = a.dropna(axis=0).values
    user1_grades = rated_movies_by_both[:, 0]
    user2_grades = rated_movies_by_both[:, 1]
    return np.corrcoef(user1_grades, user2_grades)[0, 1]
def subtract_bias(rating, mean_rating: float):
    return rating - mean_rating


def get_neighbor_rating_without_bias_per_movie(
    rf: pd.DataFrame, user: int, subject: str
):
    mean_rating = rf[user].mean()
    rating = rf.loc[subject, user]
    return subtract_bias(rating, mean_rating)
    
def get_ratings_of_neighbors(rf: pd.DataFrame, neighbors: dict, subject: str):
   
    return [
        get_neighbor_rating_without_bias_per_movie(rf, neighbor, subject)
        for neighbor in neighbors
    ]

def get_weighted_average_rating_of_neighbors(ratings: list, neighbor_distance: list):
    weighted_sum = np.array(ratings).dot(np.array(neighbor_distance))
    abs_neigbor_distance = np.abs(neighbor_distance)
    return weighted_sum / np.sum(abs_neigbor_distance)

def get_score(similarity_df:pd.DataFrame,course:str,newUser:pd.Series,rf:pd.DataFrame):
    dict=similarity_df[0][:].nlargest(2).to_dict()
    neighbors=dict
    ratings = get_ratings_of_neighbors(rf,neighbors,course)
    neighbor_distance=[]
    for n in neighbors:
        neighbor_distance.append(neighbors[n])
    avg_neighbor_rating=get_weighted_average_rating_of_neighbors(ratings, neighbor_distance)
    user_avg_rating = newUser.mean()
    return round(user_avg_rating + avg_neighbor_rating, 1)   

def get_loss(d,W,U,b,c,mu):
  n=float(len(d))
  sse=0
  for k,r in d.items():
    i,j = k
    p=W[i].dot(U[j])+b[i]+c[j]+mu
    sse += (p-r)*(p-r)
  return sse/n
         
@app.route('/')
def hello_world():
    return 'Hello! Welcome to collegeBytesAPI'

@app.route('/collabf',methods=["GET","POST"])
def func():
    suggestions = []
    if request.method=='POST':
        data=request.get_json()
        print(data)

        newUser_grades = []
      
            
        f = pd.read_csv('Student-course-matrix.csv')
        f = f.drop("StudentID", axis='columns')
        correct_courses=[]
        for x in f.columns.values:
            if x[0]==" ":
                x=x[1:]
                correct_courses.append(x)
            else:
                correct_courses.append(x)
        f.columns=correct_courses
        for course in f.columns.values:
            if course in data['grades']:
                if data['grades'][course]=="":
                    newUser_grades.append(np.nan)
                else:
                    newUser_grades.append(float(data['grades'][course]))
            else:
                newUser_grades.append(np.nan)
        newUser=pd.Series(newUser_grades)
        subjects = list(f.columns)
        students =list(f.index)
        similarity_matrix = np.array([find_correlation_between_two_users(f, user1,newUser) for user1 in students])
        similarity_df = pd.DataFrame(similarity_matrix, columns=[0],index=students)
        rf=f.transpose()
        #rated_users=rf.loc[arr["course"], :].dropna().index.values
        #print(rated_users)
        grade_scores=[]
        for course in data['courses']:
            grade_scores.append([get_score(similarity_df,course,newUser,rf),course])
        return_arr=sorted(sorted(grade_scores,key=lambda x: x[0]))
        return_arr.reverse()
        print(return_arr)
        for x in return_arr:
            suggestions.append(x[1])
        print(suggestions)

    return {
        "recommendations":suggestions
    }

@app.route('/matrixf',methods=["GET","POST"])
def function():
    suggestions = []
    if request.method=='POST':
        df=pd.read_csv('Student-course-matrix.csv')
        df = df.drop("StudentID", axis='columns')
        data=request.get_json()
        newUser_grades = []
        idx=0
        size=len(df.columns.values)
        correct_courses=[]
        for x in df.columns.values:
            if x[0]==" ":
                correct_courses.append(x[1:])
            else:
                correct_courses.append(x)
        df.columns=correct_courses
        for course in df.columns.values:
            if course in data['grades']:
                if data['grades'][course]=="":
                    newUser_grades.append(np.nan)
                else:
                    newUser_grades.append(float(data['grades'][course]))
            else:
                newUser_grades.append(np.nan)
           
        
        print(data['grades'])
        print(data['courses'])
        dict={}

        all_courses=df.columns.values

        df.loc[len(df.index)]=newUser_grades
        
        Index2course={}
        course2Index={}
        j=0
        for x in all_courses:
            Index2course[j]=x
            course2Index[x]=j
            j+=1
        col_ids=[]
        for k in range(0,size):
            col_ids.append(k)
        df.columns=col_ids
        user2course={}
        course2user={}
        usercourse2rating={}
        for j in range(0,252):
            for i in df.columns.values:
                if not pd.isna(df.loc[j][i]):
                    if j not in user2course:
                        user2course[j]=[]
                    user2course[j].append(i)
                    if i not in course2user:
                        course2user[i]=[]
                    course2user[i].append(j)
                    usercourse2rating[(j,i)]=df.loc[j][i]
        N = np.max(list(user2course.keys()))+1
        M=np.max(list(course2user.keys()))+1
        K=10
        W=np.random.randn(N,K)
        U=np.random.randn(M,K)
        b=np.zeros(N)
        c=np.zeros(M)
        mu=np.mean(list(usercourse2rating.values()))
        epochs=20
        reg=0.01
        train_losses=[]
        for e in range(epochs):
            for i in range(N):
                matrix= np.eye(K)*reg
                vector =np.zeros(K)
                bi=0
                for j in user2course[i]:
                    r=usercourse2rating[(i,j)]
                    matrix += np.outer(U[j],U[j])
                    vector += ((r-c[j]-mu-b[i])*U[j])
                    bi += (r-c[j]-mu-W[i].dot(U[j]))
                W[i] = np.linalg.solve(matrix,vector)
                b[i] = bi/(reg+(len(user2course[i])))


        for j in range(M):
            mat= np.eye(K)*reg
            vector =np.zeros(K)
            cj=0
            for i in course2user[j]:
                r=usercourse2rating[(i,j)]
                mat += np.outer(W[i],W[i])
                vector += ((r-c[j]-mu-b[i])*W[i])
                cj += (r-b[i]-mu-W[i].dot(U[j]))
            U[j] = np.linalg.solve(mat,vector)
            c[j] = cj/(reg+(len(course2user[j])))

            train_losses.append(get_loss(usercourse2rating,W,U,b,c,mu))
        
        grade_scores=[]
        for course in data['courses']:
            #if course=="IoT Analytics":
               #grade_scores.append([W[251].dot(U[course2Index[" IoT Analytics"]])+b[251]+c[course2Index[" IoT Analytics"]]+mu,course])
            #else:
            print(course2Index[course])
            grade_scores.append([W[251].dot(U[course2Index[course]])+b[251]+c[course2Index[course]]+mu,course])
            
        print(grade_scores)
        return_arr=sorted(sorted(grade_scores,key=lambda x: x[0]))
        return_arr.reverse()
        for x in return_arr:
            suggestions.append(x[1])
    return {
          "recommendations":suggestions
        }


if __name__ == '__main__':
 
    app.run(debug=True)