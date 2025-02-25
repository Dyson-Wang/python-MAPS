from influxdb import InfluxDBClient

class Database():
    def __init__(self, db_name,influxdb_host,influxdb_port):
        self.db_name = db_name
        self.host = influxdb_host
        self.port = influxdb_port
        self.conn = InfluxDBClient(host=self.host, port=self.port) # 初始化连接
        self.conn.drop_database(self.db_name) # 删除旧数据库
        self.db = self.create(self.conn,self.db_name)

    def create(self,conn,db_name):
       #Create a database
       conn.create_database(db_name) # 创建数据库
       conn.switch_database(db_name) # 切换数据库
       return conn

    def insert(self,json_body):
        # Insert data in to database
        self.db.write_points(json_body)
   
    def delete(self, query):
        # Delete data from database
        self.db.drop_measurement(query)
    
    def delete_measurement(self, query):
        self.db.query(query)

    def select(self,query):
        result =  self.db.query(query)
        return result
