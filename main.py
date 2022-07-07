from flask import Flask,render_template,request
import pickle

app=Flask(__name__)

file=open('model.pkl','rb')
clf=pickle.load(file)
file.close()
@app.route('/',methods=['GET','POST'])
def hello_world():
    if request.method=='POST':
        mydict=request.form
        fever=int(mydict['fever'])
        age=int(mydict['age'])
        pain=int(mydict['pain'])
        runnynose=int(mydict['runnynose'])
        diffbreath=int(mydict['diffbreath'])
        input=[fever,pain,age,runnynose,diffbreath]
        prob=clf.predict_proba([input])[0][1]
        print(prob)
        return render_template('show.html',inf=prob)
    return render_template('index.html')
    # return 'hello'
if __name__ == '__main__':
    app.run(debug=True)