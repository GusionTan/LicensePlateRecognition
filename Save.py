import pymssql

class Conn():
    def __init__(self):
         self.conn = pymssql.connect(host='localhost',database='Car',user='sa',password='asd123',charset='utf8')

    def Insert(self,str):
        cursor = self.conn.cursor()
        sql_insert = "insert into CarList values(getdate(),'%s')"%(str)
        try:
            cursor.execute(sql_insert)
            self.conn.commit()

        except Exception as e:
                    print (e)
            
        self.conn.close()