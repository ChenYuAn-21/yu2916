from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

db = SQLAlchemy()


class Item(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80))
    category = db.Column(db.String(50))
    condition = db.Column(db.String(50))
    price = db.Column(db.Float)
    # 備註說明（賣家可填寫的額外資訊）
    notes = db.Column(db.Text)
    # 取貨/寄送設定
    # pickup_methods: JSON list (例: ["面交","寄送"])，儲存在 Text 欄位
    pickup_methods = db.Column(db.Text)
    # 若賣家支援寄送，shipping_type 可為 '免運'、'含運'、'運費外加'
    shipping_type = db.Column(db.String(50))
    # 運費金額（僅當 shipping_type == '運費外加' 時使用）
    shipping_fee = db.Column(db.Float)
    # 付款方式：JSON list（例: ["面交收現","貨到收現"]）
    payment_methods = db.Column(db.Text)
    campus = db.Column(db.String(20))  # 本校區 or 華夏校區
    image_path = db.Column(db.String(200))
    # 向量特徵（EfficientNet / MobileNet 等產生的 float32 向量，儲存在 BLOB 欄位）
    embedding = db.Column(db.LargeBinary, nullable=True)
    # 庫存數量（上傳時可指定，預設 1）
    quantity = db.Column(db.Integer, default=1)
    # 賣家（User id）
    seller_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    # 是否上架（True 表示可於列表看到並購買）；當庫存為 0 時會設為 False
    is_active = db.Column(db.Boolean, default=True)


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True)
    # password hash for account-based login (nullable for legacy accounts)
    password_hash = db.Column(db.String(200), nullable=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        if not self.password_hash:
            return False
        return check_password_hash(self.password_hash, password)




class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    item_id = db.Column(db.Integer, db.ForeignKey('item.id'))
    sender = db.Column(db.String(50))  # 'buyer' 或 'seller'
    msg = db.Column(db.Text)
    # 發言者對應的 user id（若為匿名買家可為 NULL）
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    # 若為私訊，conversation_id 指向 Conversation
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=True)
    # 額外欄位：圖檔路徑、已讀旗標與已讀時間
    image_path = db.Column(db.String(200), nullable=True)
    is_read = db.Column(db.Boolean, default=False)
    read_at = db.Column(db.DateTime, nullable=True)


class Order(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    total = db.Column(db.Float)
    # 聯絡與交付資訊（由買家在結帳或立即購買時填寫）
    contact_phone = db.Column(db.String(50))
    delivery_method = db.Column(db.String(100))
    pickup_location = db.Column(db.String(200))
    shipping_address = db.Column(db.Text)
    # relationship optional


class OrderItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, db.ForeignKey('order.id'))
    item_id = db.Column(db.Integer, db.ForeignKey('item.id'))
    price = db.Column(db.Float)
    quantity = db.Column(db.Integer, default=1)


class CartEntry(db.Model):
    """紀錄每一個購物車（以 cart_token 表示）所包含的商品，
    用來計算有多少不同使用者/購物車把該商品加入（先搶先贏的顯示）。
    cart_token: 由 session 產生並存在 session['cart_token']。
    user_id: 若使用者登入則會存使用者 id，否則為 NULL。
    """
    id = db.Column(db.Integer, primary_key=True)
    cart_token = db.Column(db.String(64), index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    item_id = db.Column(db.Integer, db.ForeignKey('item.id'))
    quantity = db.Column(db.Integer, default=1)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Conversation(db.Model):
    """私訊會話（雙方：user1 / user2），可選關聯 item_id 表示該會話與某個商品相關。"""
    id = db.Column(db.Integer, primary_key=True)
    user1_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    user2_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    item_id = db.Column(db.Integer, db.ForeignKey('item.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
