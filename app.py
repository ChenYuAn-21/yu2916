from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session, g
from flask_socketio import SocketIO, join_room, emit
from flask_migrate import Migrate
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
import os
from datetime import datetime, timezone, timedelta
import sqlite3
from sqlalchemy.exc import OperationalError
import json

import uuid
from sqlalchemy import func
from models import db, Item, Message, User, Order, OrderItem, CartEntry, Conversation  # 引入購物與會話模型
import uuid
from sqlalchemy import func

# --- TensorFlow / Keras embedding model (EfficientNetB0 -> 1280-d) ---
try:
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
    _tf_available = True
except Exception as e:
    print('TensorFlow not available:', e)
    tf = None
    EfficientNetB0 = None
    eff_preprocess = None
    _tf_available = False

_embedding_model = None
_embedding_input_size = (224, 224)

def ensure_embedding_model():
    global _embedding_model
    if _embedding_model is None and _tf_available:
        base = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg', input_shape=(_embedding_input_size[0], _embedding_input_size[1], 3))
        _embedding_model = base
        print('Embedding model loaded (EfficientNetB0)')

def compute_image_embedding_from_bytes(image_bytes):
    """Return a float32 numpy vector (1280,) normalized L2 from image bytes."""
    # Lazy-import heavy numerical/image libraries to avoid import-time failures
    if not _tf_available:
        raise RuntimeError('TensorFlow not available')
    try:
        import io
        from PIL import Image
        import numpy as np
    except Exception as e:
        raise RuntimeError('Required imaging/numeric libraries not available: %s' % e)

    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(_embedding_input_size)
    arr = np.array(img).astype('float32')
    arr = np.expand_dims(arr, axis=0)
    arr = eff_preprocess(arr)
    emb = _embedding_model.predict(arr)
    emb = np.asarray(emb).reshape(-1)
    # normalize
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb.astype('float32')

# -----------------------------
# 初始化 Flask
# -----------------------------
app = Flask(__name__)
# 使用絕對路徑避免 working directory 差異導致 DB 檔案建立在不預期位置
DB_FILE = os.path.join(os.path.dirname(__file__), 'items.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{DB_FILE}"
app.config['SECRET_KEY'] = 'secret!'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
socketio = SocketIO(app)
db.init_app(app)
# Flask-Migrate & Flask-Login
migrate = Migrate(app, db)
login_manager = LoginManager(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    try:
        return User.query.get(int(user_id))
    except Exception:
        return None

# 確保在任何啟動方式（包括 flask run）下，資料表都會被建立
def ensure_schema():
    """檢查 SQLite schema，若缺少 user table 或欄位則嘗試建立/新增。（開發用）"""
    db_path = DB_FILE
    if not os.path.exists(db_path):
        return
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        # 建立 user table（若不存在）
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user'")
        if not cur.fetchone():
            cur.execute('CREATE TABLE user (id INTEGER PRIMARY KEY, name VARCHAR(80) UNIQUE)')

        # 檢查 item 是否有 seller_id 欄位，若無則新增
        cur.execute("PRAGMA table_info('item')")
        cols = [r[1] for r in cur.fetchall()]
        if 'seller_id' not in cols:
            try:
                cur.execute('ALTER TABLE item ADD COLUMN seller_id INTEGER')
            except Exception as e:
                print('無法新增 item.seller_id 欄位:', e)

        # 檢查 item 的其他新欄位，開發用嘗試新增（notes, pickup_methods, shipping_type, shipping_fee, payment_methods）
        cur.execute("PRAGMA table_info('item')")
        cols = [r[1] for r in cur.fetchall()]
        if 'notes' not in cols:
            try:
                cur.execute("ALTER TABLE item ADD COLUMN notes TEXT")
            except Exception as e:
                print('無法新增 item.notes 欄位:', e)
        if 'pickup_methods' not in cols:
            try:
                cur.execute("ALTER TABLE item ADD COLUMN pickup_methods TEXT")
            except Exception as e:
                print('無法新增 item.pickup_methods 欄位:', e)
        if 'shipping_type' not in cols:
            try:
                cur.execute("ALTER TABLE item ADD COLUMN shipping_type VARCHAR(50)")
            except Exception as e:
                print('無法新增 item.shipping_type 欄位:', e)
        if 'shipping_fee' not in cols:
            try:
                cur.execute("ALTER TABLE item ADD COLUMN shipping_fee FLOAT")
            except Exception as e:
                print('無法新增 item.shipping_fee 欄位:', e)
        if 'quantity' not in cols:
            try:
                # 新增數量欄位，預設為 1
                cur.execute("ALTER TABLE item ADD COLUMN quantity INTEGER DEFAULT 1")
            except Exception as e:
                print('無法新增 item.quantity 欄位:', e)
        if 'is_active' not in cols:
            try:
                # SQLite 沒有 boolean 型別，使用 INTEGER 0/1
                cur.execute("ALTER TABLE item ADD COLUMN is_active INTEGER DEFAULT 1")
            except Exception as e:
                print('無法新增 item.is_active 欄位:', e)
        if 'payment_methods' not in cols:
            try:
                cur.execute("ALTER TABLE item ADD COLUMN payment_methods TEXT")
            except Exception as e:
                print('無法新增 item.payment_methods 欄位:', e)

        # 檢查 message 是否有 user_id 欄位，若無則新增
        cur.execute("PRAGMA table_info('message')")
        cols = [r[1] for r in cur.fetchall()]
        if 'user_id' not in cols:
            try:
                cur.execute('ALTER TABLE message ADD COLUMN user_id INTEGER')
            except Exception as e:
                print('無法新增 message.user_id 欄位:', e)
        # 檢查 message 是否有 conversation_id, image_path, is_read, read_at 欄位
        cur.execute("PRAGMA table_info('message')")
        mcols = [r[1] for r in cur.fetchall()]
        if 'conversation_id' not in mcols:
            try:
                cur.execute("ALTER TABLE message ADD COLUMN conversation_id INTEGER")
            except Exception as e:
                print('無法新增 message.conversation_id 欄位:', e)
        if 'image_path' not in mcols:
            try:
                cur.execute("ALTER TABLE message ADD COLUMN image_path VARCHAR(200)")
            except Exception as e:
                print('無法新增 message.image_path 欄位:', e)
        if 'is_read' not in mcols:
            try:
                cur.execute("ALTER TABLE message ADD COLUMN is_read INTEGER DEFAULT 0")
            except Exception as e:
                print('無法新增 message.is_read 欄位:', e)
        if 'read_at' not in mcols:
            try:
                cur.execute("ALTER TABLE message ADD COLUMN read_at DATETIME")
            except Exception as e:
                print('無法新增 message.read_at 欄位:', e)

        # 檢查 user 是否有 password_hash 欄位
        cur.execute("PRAGMA table_info('user')")
        ucols = [r[1] for r in cur.fetchall()]
        if 'password_hash' not in ucols:
            try:
                cur.execute("ALTER TABLE user ADD COLUMN password_hash VARCHAR(200)")
            except Exception as e:
                print('無法新增 user.password_hash 欄位:', e)

        # 新增 conversation 表（若不存在）
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='conversation'")
        if not cur.fetchone():
            try:
                cur.execute('CREATE TABLE conversation (id INTEGER PRIMARY KEY, user1_id INTEGER, user2_id INTEGER, item_id INTEGER, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)')
            except Exception as e:
                print('無法建立 conversation 表:', e)

        # 檢查 message 是否有 conversation_id 欄位，若無則新增（開發用）
        cur.execute("PRAGMA table_info('message')")
        mcols = [r[1] for r in cur.fetchall()]
        if 'conversation_id' not in mcols:
            try:
                cur.execute("ALTER TABLE message ADD COLUMN conversation_id INTEGER")
            except Exception as e:
                print('無法新增 message.conversation_id 欄位:', e)

        # 檢查 order table 是否存在需要的欄位（開發用 ALTER 嘗試）
        try:
            cur.execute("PRAGMA table_info('order')")
            order_cols = [r[1] for r in cur.fetchall()]
            if 'contact_phone' not in order_cols:
                try:
                    cur.execute("ALTER TABLE 'order' ADD COLUMN contact_phone VARCHAR(50)")
                except Exception as e:
                    print('無法新增 order.contact_phone 欄位:', e)
            if 'delivery_method' not in order_cols:
                try:
                    cur.execute("ALTER TABLE 'order' ADD COLUMN delivery_method VARCHAR(100)")
                except Exception as e:
                    print('無法新增 order.delivery_method 欄位:', e)
            if 'pickup_location' not in order_cols:
                try:
                    cur.execute("ALTER TABLE 'order' ADD COLUMN pickup_location VARCHAR(200)")
                except Exception as e:
                    print('無法新增 order.pickup_location 欄位:', e)
            if 'shipping_address' not in order_cols:
                try:
                    cur.execute("ALTER TABLE 'order' ADD COLUMN shipping_address TEXT")
                except Exception as e:
                    print('無法新增 order.shipping_address 欄位:', e)
        except Exception as e:
            print('檢查或建立 order 欄位時發生錯誤:', e)

        conn.commit()
        # 檢查 item.embedding 欄位
        cur.execute("PRAGMA table_info('item')")
        cols_after = [r[1] for r in cur.fetchall()]
        if 'embedding' not in cols_after:
            try:
                cur.execute("ALTER TABLE item ADD COLUMN embedding BLOB")
            except Exception as e:
                print('無法新增 item.embedding 欄位:', e)
    except Exception as e:
        print('ensure_schema 發生錯誤:', e)
    finally:
        try:
            if conn:
                conn.close()
        except:
            pass

try:
    with app.app_context():
        db.create_all()
        ensure_schema()
except Exception as e:
    print("建立資料表時發生錯誤：", e)

# Jinja 過濾器：將儲存在 DB 的 UTC datetime（naive 或 aware）轉為 GMT+8 並格式化字串
def _fmt_gmt8(dt, fmt="%Y-%m-%d %H:%M:%S"):
    if not dt:
        return ''
    try:
        # 若為 naive，視為 UTC
        if getattr(dt, 'tzinfo', None) is None:
            aware = dt.replace(tzinfo=timezone.utc)
        else:
            aware = dt
        target = timezone(timedelta(hours=8))
        local = aware.astimezone(target)
        return local.strftime(fmt)
    except Exception:
        try:
            return dt.strftime(fmt)
        except Exception:
            return str(dt)

app.jinja_env.filters['gmt8'] = _fmt_gmt8

# -----------------------------
# 上傳允許格式
# -----------------------------
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'jfif'}
def allowed_file(filename):
    return '.' in filename and '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# -----------------------------
# 首頁篩選
# -----------------------------
def apply_filters(query, req):
    keyword = req.args.get('q', '')
    category = req.args.get('category', '')
    campus = req.args.get('campus', '')
    # 新增價格與新舊程度篩選參數
    min_price = req.args.get('min_price')
    max_price = req.args.get('max_price')
    condition = req.args.get('condition', '')
    if keyword:
        query = query.filter(Item.name.contains(keyword))
    if category:
        query = query.filter_by(category=category)
    if campus:
        query = query.filter_by(campus=campus)
    # 處理價格範圍
    try:
        if min_price is not None and min_price != '':
            mp = float(min_price)
            query = query.filter(Item.price != None).filter(Item.price >= mp)
    except Exception:
        pass
    try:
        if max_price is not None and max_price != '':
            mp2 = float(max_price)
            query = query.filter(Item.price != None).filter(Item.price <= mp2)
    except Exception:
        pass
    # 處理新舊程度：
    # - 若選「二手」，則包含上傳表單中的良好/普通/稍舊
    # - 若選「全新」，則只包含全新
    if condition:
        if condition == '二手':
            query = query.filter(Item.condition.in_(['良好', '普通', '稍舊']))
        else:
            # 包括 '全新' 或其他精確比對值
            query = query.filter_by(condition=condition)
    return query

@app.route('/')
def index():
    # 只顯示仍上架的商品
    items = Item.query.filter_by(is_active=True)
    # 讀取目前的篩選參數，並傳給模板以便保留表單狀態
    q = request.args.get('q', '')
    category = request.args.get('category', '')
    campus = request.args.get('campus', '')
    min_price = request.args.get('min_price', '')
    max_price = request.args.get('max_price', '')
    condition = request.args.get('condition', '')
    try:
        items = apply_filters(items, request).all()
    except OperationalError as e:
        # 若因為 schema 欄位缺失導致查詢失敗，嘗試修補 schema 然後重試一次
        print('OperationalError on items query, attempting schema fix:', e)
        ensure_schema()
        items = apply_filters(items, request).all()
    # 如果使用者已登入，撈取該使用者的會話清單以便在首頁顯示連結
    convs_out = []
    try:
        from models import Conversation, Message, User
        if current_user and getattr(current_user, 'is_authenticated', False):
            convs = Conversation.query.filter((Conversation.user1_id == current_user.id) | (Conversation.user2_id == current_user.id)).order_by(Conversation.created_at.desc()).all()
            for c in convs:
                other_id = c.user1_id if c.user2_id == current_user.id else c.user2_id
                other = User.query.get(other_id)
                last = Message.query.filter_by(conversation_id=c.id).order_by(Message.timestamp.desc()).first()
                convs_out.append({'conv': c, 'other': other, 'last': last})
    except Exception:
        convs_out = []

    return render_template('index.html', items=items, q=q, category=category, campus=campus,
                           min_price=min_price, max_price=max_price, condition=condition, conversations=convs_out)


@app.context_processor
def inject_cart_distinct_count():
    """提供給模板：購物車中不重複商品的數量。
    計算方式：
    - 若 session 中有 `cart`（dict），以其 keys 為基礎
    - 也從當前 session 的 `cart_token` 與已登入使用者的 `CartEntry` 撈取，將三者的 item_id 取 union
    這樣可以保證不會因為資料在 session / DB 分散而造成計數不一致。
    """
    try:
        item_ids = set()
        # session cart 優先加入（若存在）
        cart = session.get('cart') or {}
        if isinstance(cart, dict):
            for k in cart.keys():
                try:
                    item_ids.add(int(k))
                except Exception:
                    continue

        # 加入以 cart_token 儲存的 CartEntry
        try:
            token = session.get('cart_token')
            if token:
                rows = CartEntry.query.filter_by(cart_token=token).all()
                for r in rows:
                    if r and r.item_id:
                        item_ids.add(int(r.item_id))
        except Exception:
            pass

        # 若使用者已登入，也加入以 user_id 儲存的 CartEntry
        try:
            if current_user and getattr(current_user, 'is_authenticated', False):
                rows = CartEntry.query.filter_by(user_id=current_user.id).all()
                for r in rows:
                    if r and r.item_id:
                        item_ids.add(int(r.item_id))
        except Exception:
            pass

        distinct = len(item_ids)
    except Exception:
        distinct = 0
    return dict(cart_distinct_count=distinct)


@app.context_processor
def inject_unread_conv_count():
    """提供給模板：私訊未讀總數（屬於當前使用者、且不是自己發的未讀訊息）。"""
    try:
        if current_user and getattr(current_user, 'is_authenticated', False):
            # count distinct conversations that have unread messages for current_user
            from models import Message, Conversation
            # use SQLAlchemy func to count distinct conversation_id
            cnt = db.session.query(func.count(func.distinct(Message.conversation_id))).join(Conversation, Message.conversation_id == Conversation.id).filter(
                Message.is_read == False,
                Message.user_id != current_user.id,
                ((Conversation.user1_id == current_user.id) | (Conversation.user2_id == current_user.id))
            ).scalar() or 0
            return dict(unread_conv_count=int(cnt))
    except Exception:
        pass
    return dict(unread_conv_count=0)

# -----------------------------
# 上傳二手物品
# -----------------------------
@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        name = request.form.get('name')
        category = request.form.get('category')
        campus = request.form.get('campus')
        condition = request.form.get('condition')
        price_raw = request.form.get('price')
        try:
            # 價格只需整數（不需要小數）
            price = int(float(price_raw)) if price_raw else None
        except (ValueError, TypeError):
            price = None
        file = request.files.get('image')
        filename = None
        emb_blob = None
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # compute and store embedding for this item image if model available
            try:
                ensure_embedding_model()
                with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'rb') as f:
                    b = f.read()
                emb = compute_image_embedding_from_bytes(b) if _tf_available else None
                emb_blob = emb.tobytes() if emb is not None else None
            except Exception as e:
                print('compute embedding for uploaded item failed:', e)
                emb_blob = None
        notes = request.form.get('notes')
        # 數量欄位（預設 1）
        qty_raw = request.form.get('quantity')
        try:
            quantity_val = int(qty_raw) if qty_raw not in (None, '') else 1
            if quantity_val < 1:
                quantity_val = 1
        except Exception:
            quantity_val = 1
        # 取貨與付款
        # pickup_methods 可以有多個（checkbox），Flask 返回同名欄位可用 getlist
        pickup_methods = request.form.getlist('pickup_methods')
        payment_methods = request.form.getlist('payment_methods')
        shipping_type = request.form.get('shipping_type')
        shipping_fee = request.form.get('shipping_fee')
        try:
            shipping_fee_val = float(shipping_fee) if shipping_fee not in (None, '') else None
        except Exception:
            shipping_fee_val = None
        # 決定 seller：使用已登入的 current_user（login_required 保證存在）
        seller_id = None
        if current_user and getattr(current_user, 'is_authenticated', False):
            seller_id = current_user.id
        elif session.get('user_id'):
            # 兼容舊的 session-based 切換
            seller_id = session.get('user_id')

        item = Item(name=name, category=category, campus=campus, condition=condition, price=price, image_path=filename, seller_id=seller_id, notes=notes,
            pickup_methods=json.dumps(pickup_methods) if pickup_methods else None,
                payment_methods=json.dumps(payment_methods) if payment_methods else None,
                shipping_type=shipping_type,
            shipping_fee=shipping_fee_val,
            quantity=quantity_val,
            embedding=emb_blob)
        db.session.add(item)
        db.session.commit()
        return redirect(url_for('index'))
    return render_template('upload_item.html')


# '/set_user' quick-switch route removed to avoid conflict with registration/login.
# If you need a developer-only quick-switch later, consider keeping it behind a config flag.


@app.route('/logout')
def logout():
    # logout both flask-login and session
    try:
        logout_user()
    except Exception:
        pass
    session.pop('user_id', None)
    session.pop('user_name', None)
    return redirect(request.referrer or url_for('index'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        pwd = request.form.get('password', '')
        if not name or not pwd:
            return render_template('register.html', error='請輸入名稱與密碼')
        if User.query.filter_by(name=name).first():
            return render_template('register.html', error='使用者名稱已存在')
        user = User(name=name)
        user.set_password(pwd)
        db.session.add(user)
        db.session.commit()
        login_user(user)
        session['user_id'] = user.id
        session['user_name'] = user.name
        return redirect(url_for('index'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        pwd = request.form.get('password', '')
        user = User.query.filter_by(name=name).first()
        if not user or not user.check_password(pwd):
            return render_template('login.html', error='名稱或密碼錯誤')
        login_user(user)
        session['user_id'] = user.id
        session['user_name'] = user.name
        # 如果 session 中有 cart_token，將該 token 相關的 CartEntry 賦予此 user_id（登入後合併購物車）
        try:
            token = session.get('cart_token')
            if token:
                entries = CartEntry.query.filter_by(cart_token=token).all()
                for e in entries:
                    if not e.user_id:
                        e.user_id = user.id
                        db.session.add(e)
                if entries:
                    db.session.commit()
        except Exception as e:
            db.session.rollback()
        next_url = request.args.get('next') or url_for('index')
        return redirect(next_url)
    return render_template('login.html')


# -----------------------------
# 購物車與訂單
# -----------------------------
def _get_cart():
    # cart stored in session as { item_id_str: qty }
    return session.setdefault('cart', {})


@app.route('/cart')
def view_cart():
    cart = _get_cart()
    items = []
    total = 0.0
    for sid, qty in cart.items():
        try:
            iid = int(sid)
            item = Item.query.get(iid)
            if item:
                items.append({'item': item, 'qty': int(qty)})
                total += (item.price or 0) * int(qty)
        except Exception:
            continue
    # 讀取一次性錯誤訊息（例如 checkout 驗證錯誤）
    checkout_error = None
    try:
        checkout_error = session.pop('checkout_error') if session.get('checkout_error') else None
    except Exception:
        checkout_error = None
    return render_template('cart.html', items=items, total=total, checkout_error=checkout_error)



@app.before_request
def ensure_cart_token():
    # 為每個 session 建立唯一 cart_token（用於匿名購物車辨識）
    if 'cart_token' not in session:
        session['cart_token'] = uuid.uuid4().hex


@app.route('/cart/add/<int:item_id>', methods=['POST'])
@login_required
def add_to_cart(item_id):
    try:
        qty = int(request.form.get('qty', 1))
    except Exception:
        qty = 1
    cart = _get_cart()
    key = str(item_id)

    # 檢查商品庫存，避免加入超過上架數量
    try:
        item = Item.query.get(item_id)
        available = int(item.quantity) if item and item.quantity is not None else 0
    except Exception:
        available = 0

    # 已在購物車中的數量
    current_in_cart = int(cart.get(key, 0))
    # 可再加入的數量
    allowed = max(0, available - current_in_cart)
    add_qty = min(max(0, qty), allowed)
    if add_qty > 0:
        cart[key] = current_in_cart + add_qty
        session['cart'] = cart
        # 同步寫入或更新 CartEntry
        try:
            token = session.get('cart_token')
            uid = None
            if current_user and getattr(current_user, 'is_authenticated', False):
                uid = current_user.id
            # 查詢既有 entry
            entry = CartEntry.query.filter_by(cart_token=token, item_id=item_id).first()
            if entry:
                entry.quantity = cart[key]
                if uid:
                    entry.user_id = uid
            else:
                entry = CartEntry(cart_token=token, user_id=uid, item_id=item_id, quantity=cart[key])
                db.session.add(entry)
            db.session.commit()
        except Exception:
            db.session.rollback()
    # 若 add_qty == 0，代表庫存不足或已滿，直接不加入
    # 若表單包含聯絡資料（給賣家），則建立一則 Message 以便賣家收到買家聯絡方式
    try:
        contact_lastname = request.form.get('contact_lastname')
        contact_role = request.form.get('contact_role')
        shipping_address = request.form.get('shipping_address')
        if contact_lastname or contact_role or shipping_address:
            # 建立內容：包含簡短聯絡摘要
            parts = []
            if contact_lastname:
                parts.append(f"姓氏: {contact_lastname}")
            if contact_role:
                parts.append(f"身分: {contact_role}")
            if shipping_address:
                parts.append(f"寄送地址: {shipping_address}")
            msg_text = "; ".join(parts)
            # 找到該物品與賣家
            try:
                item = Item.query.get(item_id)
                seller_id = item.seller_id if item else None
            except Exception:
                seller_id = None
            # 儲存為 buyer 發給賣家的訊息（若有賣家）
            try:
                m = None
                if seller_id:
                    m = Message(item_id=item_id, sender='buyer', msg=msg_text, user_id=session.get('user_id'))
                    db.session.add(m)
                    db.session.commit()
            except Exception:
                db.session.rollback()
    except Exception:
        pass
    return redirect(request.referrer or url_for('view_cart'))


@app.route('/cart/remove/<int:item_id>', methods=['POST'])
def remove_from_cart(item_id):
    cart = _get_cart()
    key = str(item_id)
    if key in cart:
        cart.pop(key)
        session['cart'] = cart
    # 同步刪除 CartEntry
    try:
        token = session.get('cart_token')
        # 刪除以 cart_token 儲存的 entry
        if token:
            CartEntry.query.filter_by(cart_token=token, item_id=item_id).delete()
        # 若使用者已登入，也刪除以 user_id 關聯的 entry（確保不會遺留）
        if current_user and getattr(current_user, 'is_authenticated', False):
            try:
                CartEntry.query.filter_by(user_id=current_user.id, item_id=item_id).delete()
            except Exception:
                pass
        db.session.commit()
    except Exception:
        db.session.rollback()
    return redirect(request.referrer or url_for('view_cart'))


@app.route('/cart/cleanup', methods=['GET'])
def cart_cleanup():
    """One-off helper: clear CartEntry rows associated with current session's cart_token
    and (if logged in) the current user's CartEntry rows, then clear session['cart'].
    Use this to remove stale DB entries that cause the nav badge to show incorrect counts.
    This is intended for local/dev use; remove or protect after use if you prefer.
    """
    try:
        token = session.get('cart_token')
        uid = None
        try:
            if current_user and getattr(current_user, 'is_authenticated', False):
                uid = current_user.id
        except Exception:
            uid = session.get('user_id')

        removed = 0
        if token:
            res = CartEntry.query.filter_by(cart_token=token).delete()
            removed += (res or 0)
        if uid:
            res2 = CartEntry.query.filter_by(user_id=uid).delete()
            removed += (res2 or 0)
        db.session.commit()
        # clear session cart
        session['cart'] = {}
        return redirect(url_for('view_cart'))
    except Exception as e:
        db.session.rollback()
        return f"cart cleanup failed: {e}", 500


@app.route('/checkout', methods=['POST'])
@login_required
def checkout():
    cart = _get_cart()
    if not cart:
        return redirect(url_for('view_cart'))

    # 讀取表單聯絡資料並驗證
    contact_lastname = request.form.get('contact_lastname')
    contact_role = request.form.get('contact_role')
    contact_phone = request.form.get('contact_phone')
    delivery_method = request.form.get('delivery_method')
    shipping_address = request.form.get('shipping_address')

    if not contact_phone:
        session['checkout_error'] = '請輸入聯絡手機，方便賣家聯絡。'
        return redirect(url_for('view_cart'))
    if not delivery_method:
        session['checkout_error'] = '請選擇取貨或寄送方式。'
        return redirect(url_for('view_cart'))
    if delivery_method == 'shipping' and (not shipping_address or shipping_address.strip() == ''):
        session['checkout_error'] = '選擇寄送時請輸入寄送地址。'
        return redirect(url_for('view_cart'))

    total = 0.0
    # 先檢查所有項目的庫存是否足夠
    try:
        for sid, qty in cart.items():
            iid = int(sid)
            req_qty = int(qty)
            item = Item.query.get(iid)
            if not item:
                raise Exception(f"商品 {iid} 不存在")
            available = int(item.quantity) if item.quantity is not None else 0
            if req_qty > available:
                raise Exception(f"商品 {item.name} 庫存不足（需要 {req_qty}，僅剩 {available}）")
    except Exception as e:
        session['checkout_error'] = f"結帳前庫存檢查失敗: {e}"
        return redirect(url_for('view_cart'))

    # 若都足夠，建立訂單並扣庫存（同一交易）
    try:
        order = Order(user_id=current_user.id, total=0.0,
                      contact_phone=contact_phone,
                      delivery_method=delivery_method,
                      shipping_address=shipping_address)
        db.session.add(order)
        db.session.flush()
        order_items = []
        for sid, qty in cart.items():
            iid = int(sid)
            req_qty = int(qty)
            item = Item.query.get(iid)
            price = item.price or 0
            oi = OrderItem(order_id=order.id, item_id=item.id, price=price, quantity=req_qty)
            db.session.add(oi)
            total += price * req_qty
            order_items.append({'item': item, 'qty': req_qty})
            # 扣庫存
            item.quantity = int(item.quantity) - req_qty if item.quantity is not None else 0
            if item.quantity <= 0:
                item.quantity = 0
                item.is_active = False
            db.session.add(item)

        order.total = total

        # 儲存聯絡資訊為訊息給賣家（每個 item）
        parts = []
        if contact_lastname:
            parts.append(f"姓氏: {contact_lastname}")
        if contact_role:
            parts.append(f"身分: {contact_role}")
        parts.append(f"手機: {contact_phone}")
        parts.append(f"取貨方式: {delivery_method}")
        if shipping_address:
            parts.append(f"寄送地址: {shipping_address}")
        msg_text = "; ".join(parts)
        for sid, qty in cart.items():
            iid = int(sid)
            m = Message(item_id=iid, sender='buyer', msg=msg_text, user_id=current_user.id)
            db.session.add(m)

        db.session.commit()
        # clear cart
        session['cart'] = {}
        # 顯示訂單成功頁面
        return render_template('order_success.html', order=order, order_items=order_items)
    except Exception as e:
        db.session.rollback()
        session['checkout_error'] = f"結帳失敗: {e}"
        return redirect(url_for('view_cart'))


@app.route('/buy_now/<int:item_id>', methods=['POST'])
@login_required
def buy_now(item_id):
    """
    立即購買單一商品：會建立一筆 Order 與 OrderItem（若使用者已登入則關聯 user），
    並將買家留下的聯絡方式儲存為一則 Message 讓賣家可見。
    表單欄位：contact_lastname, contact_role, shipping_address
    """
    item = Item.query.get_or_404(item_id)
    # 讀取聯絡資料
    contact_lastname = request.form.get('contact_lastname')
    contact_role = request.form.get('contact_role')
    shipping_address = request.form.get('shipping_address')
    contact_phone = request.form.get('contact_phone')
    delivery_method = request.form.get('delivery_method') or request.form.get('pickup_option') or request.form.get('shipping_option')

    try:
        uid = None
        if current_user and getattr(current_user, 'is_authenticated', False):
            uid = current_user.id
        elif session.get('user_id'):
            uid = session.get('user_id')

        # 驗證必填欄位：phone 必填；若選寄送，shipping_address 必填；delivery_method 必填
        if not contact_phone:
            session['buy_error'] = '請輸入手機號碼，方便賣家聯絡。'
            return redirect(url_for('item_detail', item_id=item.id))
        if not delivery_method:
            session['buy_error'] = '請選擇取貨或寄送方式。'
            return redirect(url_for('item_detail', item_id=item.id))
        shipping_address = request.form.get('shipping_address')
        if delivery_method == 'shipping' and not shipping_address:
            session['buy_error'] = '選擇寄送時請輸入寄送地址。'
            return redirect(url_for('item_detail', item_id=item.id))

        # 檢查庫存，立即購買預設為 1 件
        available = int(item.quantity) if item and item.quantity is not None else 0
        if available < 1:
            session['buy_error'] = '庫存不足，無法購買。'
            return redirect(url_for('item_detail', item_id=item.id))

        # 建立訂單並在同一交易中扣除庫存
        order = Order(user_id=uid, total=0.0,
                  contact_phone=contact_phone if 'contact_phone' in locals() else None,
                  delivery_method=delivery_method if 'delivery_method' in locals() else None,
                  shipping_address=shipping_address if 'shipping_address' in locals() else None)
        db.session.add(order)
        db.session.flush()

        price = item.price or 0
        oi = OrderItem(order_id=order.id, item_id=item.id, price=price, quantity=1)
        db.session.add(oi)
        order.total = price

        # 扣庫存
        try:
            item.quantity = int(item.quantity) - 1 if item.quantity is not None else 0
            if item.quantity <= 0:
                item.quantity = 0
                item.is_active = False
            db.session.add(item)
        except Exception as e:
            db.session.rollback()
            return f"庫存更新失敗: {e}", 500

        # 若有聯絡資料或購買資訊，儲存為訊息給賣家（包含手機、配送方式與寄送地址）
        parts = []
        if contact_lastname:
            parts.append(f"姓氏: {contact_lastname}")
        if contact_role:
            parts.append(f"身分: {contact_role}")
        if contact_phone:
            parts.append(f"手機: {contact_phone}")
        if delivery_method:
            parts.append(f"取貨方式: {delivery_method}")
        if shipping_address:
            parts.append(f"寄送地址: {shipping_address}")
        if parts:
            msg_text = "; ".join(parts)
            m = Message(item_id=item.id, sender='buyer', msg=msg_text, user_id=uid)
            db.session.add(m)

        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return f"購買過程發生錯誤: {e}", 500

    # 若使用者已登入，導向訂單頁面；否則導回商品頁並告知已送出聯絡
    if uid:
        return redirect(url_for('orders'))
    else:
        return redirect(url_for('item_detail', item_id=item.id))


@app.route('/orders')
@login_required
def orders():
    user_orders = Order.query.filter_by(user_id=current_user.id).order_by(Order.created_at.desc()).all()
    # fetch items per order
    results = []
    for o in user_orders:
        oitems = OrderItem.query.filter_by(order_id=o.id).all()
        detailed = []
        for oi in oitems:
            it = Item.query.get(oi.item_id)
            detailed.append({'order_item': oi, 'item': it})
        # use key name that doesn't shadow dict.methods (avoid 'items' key)
        results.append({'order': o, 'order_items': detailed})
    return render_template('orders.html', orders=results)


@app.route('/conversations')
@login_required
def conversations_list():
    # 列出與當前使用者有關的會話
    convs = Conversation.query.filter((Conversation.user1_id == current_user.id) | (Conversation.user2_id == current_user.id)).order_by(Conversation.created_at.desc()).all()
    # fetch last message preview
    out = []
    for c in convs:
        last = Message.query.filter_by(conversation_id=c.id).order_by(Message.timestamp.desc()).first()
        other_id = c.user1_id if c.user2_id == current_user.id else c.user2_id
        other = User.query.get(other_id)
        # 計算此會話中，屬於對方且未讀的訊息數
        try:
            unread_count = Message.query.filter_by(conversation_id=c.id, is_read=False).filter(Message.user_id != current_user.id).count()
        except Exception:
            unread_count = 0
        out.append({'conv': c, 'other': other, 'last': last, 'unread_count': unread_count})
    return render_template('conversations.html', conversations=out)


@app.route('/conversations/start/<int:seller_id>')
@login_required
def conversations_start(seller_id):
    # optional item_id in querystring
    item_id = request.args.get('item_id')
    if seller_id == current_user.id:
        return redirect(url_for('item_detail', item_id=item_id) if item_id else url_for('index'))
    # find existing conversation between these two users (optionally same item)
    c = Conversation.query.filter(
        ((Conversation.user1_id == current_user.id) & (Conversation.user2_id == seller_id)) |
        ((Conversation.user1_id == seller_id) & (Conversation.user2_id == current_user.id))
    ).first()
    if not c:
        c = Conversation(user1_id=current_user.id, user2_id=seller_id, item_id=item_id)
        db.session.add(c)
        db.session.commit()
    return redirect(url_for('conversation_view', conv_id=c.id))


@app.route('/conversations/<int:conv_id>', methods=['GET', 'POST'])
@login_required
def conversation_view(conv_id):
    c = Conversation.query.get_or_404(conv_id)
    if current_user.id not in (c.user1_id, c.user2_id):
        return "無權限存取此會話", 403
    if request.method == 'POST':
        text = request.form.get('msg')
        if text and text.strip():
            m = Message(conversation_id=c.id, sender='user', msg=text.strip(), user_id=current_user.id)
            db.session.add(m)
            db.session.commit()
        return redirect(url_for('conversation_view', conv_id=conv_id))
    # Mark unread messages (sent by the other party) as read when viewing the conversation
    try:
        uid = current_user.id if (current_user and getattr(current_user, 'is_authenticated', False)) else None
        if uid:
            unread_msgs = Message.query.filter_by(conversation_id=c.id, is_read=False).filter(Message.user_id != uid).all()
            for um in unread_msgs:
                um.is_read = True
                um.read_at = datetime.utcnow()
                db.session.add(um)
            if unread_msgs:
                db.session.commit()
    except Exception as e:
        db.session.rollback()
        print('marking unread as read in conversation_view failed:', e)

    msgs = Message.query.filter_by(conversation_id=c.id).order_by(Message.timestamp.asc()).all()
    other_id = c.user1_id if c.user2_id == current_user.id else c.user2_id
    other = User.query.get(other_id)
    return render_template('conversation.html', conv=c, other=other, messages=msgs)

# -----------------------------
# 送出圖片
# -----------------------------
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # guard: if filename is None or empty, return 404
    if not filename:
        return ("Not Found", 404)
    safe_name = os.path.basename(filename)
    upload_dir = app.config['UPLOAD_FOLDER']
    requested = os.path.join(upload_dir, safe_name)
    if os.path.exists(requested):
        return send_from_directory(upload_dir, safe_name)
    name, ext = os.path.splitext(safe_name)
    if not ext:
        for e in ('.jpg', '.jpeg', '.png', '.gif'):
            candidate = name + e
            if os.path.exists(os.path.join(upload_dir, candidate)):
                return send_from_directory(upload_dir, candidate)
    return ("Not Found", 404)


@app.route('/upload_message_image', methods=['POST'])
@login_required
def upload_message_image():
    file = request.files.get('image')
    if not file:
        return {'error': 'no file'}, 400
    if not allowed_file(file.filename):
        return {'error': 'invalid extension'}, 400
    filename = secure_filename(file.filename)
    # to avoid name collisions, prefix with uuid
    name = f"msg_{uuid.uuid4().hex}_{filename}"
    path = os.path.join(app.config['UPLOAD_FOLDER'], name)
    file.save(path)
    url = url_for('uploaded_file', filename=name)
    print(f"upload_message_image: saved file {name} at {path}, url={url}")
    # 嘗試為所上傳的 image 計算 embedding（如果可用），但若失敗不影響上傳
    try:
        ensure_embedding_model()
        with open(path, 'rb') as f:
            b = f.read()
        emb = compute_image_embedding_from_bytes(b) if _tf_available else None
        if emb is not None:
            # 若此檔案是為某個 item 上傳（通常由上傳商品流程），需在該流程儲存 embedding
            # 這裡僅回傳 embedding bytes 給前端（client 若需要可回傳 filename）
            pass
    except Exception as e:
        print('embedding compute for upload_message_image failed:', e)
    # 回傳檔名與可存取 URL
    return {'url': url, 'filename': name}


@app.route('/search_by_image', methods=['POST'])
def search_by_image():
    """接收上傳圖片，計算 embedding，並用 cosine similarity 找出最相似的 items（回傳 top N）。"""
    if not _tf_available:
        return {'error': 'TensorFlow not available on server'}, 500
    f = request.files.get('image')
    if not f:
        return {'error': 'no file'}, 400
    try:
        b = f.read()
        ensure_embedding_model()
        qvec = compute_image_embedding_from_bytes(b)
    except Exception as e:
        return {'error': str(e)}, 500

    # fetch items with embedding
    candidates = Item.query.filter(Item.embedding != None, Item.image_path != None).all()
    sims = []
    for it in candidates:
        try:
            emb_blob = it.embedding
            if not emb_blob:
                continue
            import numpy as np
            arr = np.frombuffer(emb_blob, dtype=np.float32)
            # ensure normalized
            if arr.size != qvec.size:
                continue
            # cosine similarity since vectors are normalized, dot product suffices
            score = float(np.dot(qvec, arr))
            sims.append((score, it))
        except Exception:
            continue
    sims.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, it in sims[:20]:
        results.append({'item_id': it.id, 'score': score, 'name': it.name, 'image_url': url_for('uploaded_file', filename=it.image_path)})
    return {'results': results}


@app.route('/__schema')
def _schema():
    """臨時 debug：顯示 sqlite tables 欄位資訊，僅供本機開發檢查用"""
    db_path = DB_FILE
    if not os.path.exists(db_path):
        return "no db file"
    out = {}
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        for tbl in ('item', 'message', 'user'):
            try:
                cur.execute(f"PRAGMA table_info('{tbl}')")
                cols = [r[1] for r in cur.fetchall()]
            except Exception:
                cols = None
            out[tbl] = cols
        # 檢查 order table 欄位
        try:
            cur.execute("PRAGMA table_info('order')")
            ocols = [r[1] for r in cur.fetchall()]
        except Exception:
            ocols = None
        out['order'] = ocols
    except Exception as e:
        return f"schema check error: {e}"
    finally:
        try:
            conn.close()
        except:
            pass
    return '<br>'.join([f"{k}: {v}" for k, v in out.items()])


@app.route('/__clear_messages', methods=['GET', 'POST'])
def _clear_messages():
    """臨時開發路由：刪除聊天室訊息。
    - 若提供 query param `item_id`，只刪該物品的訊息；否則刪全部訊息。
    - 僅在 app.debug 為 True 時允許執行（避免在 production 被誤用）。
    """
    if not app.debug:
        return "Not allowed", 403
    item_id = request.args.get('item_id')
    try:
        if item_id:
            n = Message.query.filter_by(item_id=int(item_id)).delete()
        else:
            n = Message.query.delete()
        db.session.commit()
        return f"deleted {n} messages"
    except Exception as e:
        db.session.rollback()
        return f"error: {e}", 500

# -----------------------------
# 物品詳情
# -----------------------------
@app.route("/item/<int:item_id>")
def item_detail(item_id):
    item = Item.query.get_or_404(item_id)
    seller = None
    if item.seller_id:
        seller = User.query.get(item.seller_id)
    # 解析 JSON 欄位傳給模板，避免依賴 template filter
    pickup_methods = []
    payment_methods = []
    try:
        if item.pickup_methods:
            pickup_methods = json.loads(item.pickup_methods)
    except Exception:
        pickup_methods = []
    try:
        if item.payment_methods:
            payment_methods = json.loads(item.payment_methods)
    except Exception:
        payment_methods = []
    # 計算有多少不同購物車把此商品加入（表示先搶先贏的參考）
    try:
        carts_count = CartEntry.query.filter_by(item_id=item_id).count()
    except Exception:
        carts_count = 0
    # 讀取一次性錯誤（例如 buy_now 的驗證錯誤），取出後移除
    form_error = None
    try:
        form_error = session.pop('buy_error') if session.get('buy_error') else None
    except Exception:
        form_error = None
    return render_template("item_detail.html", item=item, seller=seller, pickup_methods=pickup_methods, payment_methods=payment_methods, carts_count=carts_count, form_error=form_error)


@app.route('/item/<int:item_id>/edit', methods=['GET', 'POST'])
def edit_item(item_id):
    item = Item.query.get_or_404(item_id)
    current_user_id = session.get('user_id')
    # 只有賣家可以編輯
    if not current_user_id or not item.seller_id or int(current_user_id) != int(item.seller_id):
        return "未授權", 403

    if request.method == 'POST':
        # 取得欄位並更新
        item.name = request.form.get('name')
        item.category = request.form.get('category')
        item.condition = request.form.get('condition')
        price_raw = request.form.get('price')
        try:
            # 價格只需整數（不需要小數）
            item.price = int(float(price_raw)) if price_raw else None
        except (ValueError, TypeError):
            item.price = None
        # 數量
        qty_raw = request.form.get('quantity')
        try:
            item.quantity = int(qty_raw) if qty_raw not in (None, '') else 1
            if item.quantity < 1:
                item.quantity = 1
        except Exception:
            item.quantity = 1
        item.campus = request.form.get('campus')

        # 備註欄位
        item.notes = request.form.get('notes')
        # 取貨 / 付款
        pickup_methods = request.form.getlist('pickup_methods')
        payment_methods = request.form.getlist('payment_methods')
        item.pickup_methods = json.dumps(pickup_methods) if pickup_methods else None
        item.payment_methods = json.dumps(payment_methods) if payment_methods else None
        item.shipping_type = request.form.get('shipping_type')
        try:
            item.shipping_fee = float(request.form.get('shipping_fee')) if request.form.get('shipping_fee') not in (None, '') else None
        except Exception:
            item.shipping_fee = None

        # 處理圖片更新（若有上傳新檔）
        file = request.files.get('image')
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            item.image_path = filename

        db.session.commit()
        return redirect(url_for('item_detail', item_id=item.id))

    # GET: 呈現表單，預填資料
    # 解析 JSON 欄位供模板使用
    pickup_methods = []
    payment_methods = []
    try:
        if item.pickup_methods:
            pickup_methods = json.loads(item.pickup_methods)
    except Exception:
        pickup_methods = []
    try:
        if item.payment_methods:
            payment_methods = json.loads(item.payment_methods)
    except Exception:
        payment_methods = []
    return render_template('edit_item.html', item=item, pickup_methods=pickup_methods, payment_methods=payment_methods)


@app.route('/item/<int:item_id>/unpublish', methods=['POST'])
def unpublish_item(item_id):
    """賣家可用：將商品標記為未上架（is_active=False）。"""
    item = Item.query.get_or_404(item_id)
    current_user_id = session.get('user_id')
    # 只有賣家可以下架
    if not current_user_id or not item.seller_id or int(current_user_id) != int(item.seller_id):
        return "未授權", 403

    try:
        item.is_active = False
        db.session.commit()
    except Exception:
        db.session.rollback()
        # 若更新失敗，回到編輯頁並顯示錯誤
        session['buy_error'] = '下架失敗，請稍後再試。'
        return redirect(url_for('edit_item', item_id=item.id))

    # 成功下架，導回商品頁面或賣家管理頁面
    try:
        session['buy_error'] = None
    except Exception:
        pass
    return redirect(url_for('item_detail', item_id=item.id))


@app.route('/item/<int:item_id>/toggle_active', methods=['POST'])
def toggle_item_active(item_id):
    """賣家可用：切換商品上架狀態（is_active True/False）。"""
    item = Item.query.get_or_404(item_id)
    current_user_id = session.get('user_id')
    # 只有賣家可以切換上架狀態
    if not current_user_id or not item.seller_id or int(current_user_id) != int(item.seller_id):
        return "未授權", 403

    try:
        item.is_active = not bool(item.is_active)
        db.session.commit()
    except Exception:
        db.session.rollback()
        session['buy_error'] = '更新上架狀態失敗，請稍後再試。'
        return redirect(url_for('edit_item', item_id=item.id))

    return redirect(url_for('item_detail', item_id=item.id))


@app.route('/seller')
def seller_center():
    """賣家管理頁：列出該會員所有刊登過的商品，顯示上下架狀態與銷售統計／狀態。"""
    current_user_id = session.get('user_id')
    if not current_user_id:
        return redirect(url_for('login'))

    try:
        items = Item.query.filter_by(seller_id=current_user_id).all()
    except Exception:
        items = []

    # 計算每個商品的訂單數與銷售數量
    items_info = []
    for it in items:
        try:
            total_qty = db.session.query(func.coalesce(func.sum(OrderItem.quantity), 0)).filter(OrderItem.item_id == it.id).scalar() or 0
        except Exception:
            total_qty = 0

        if total_qty == 0:
            sales_status = '無人下單'
        else:
            sales_status = f'有人下單（{int(total_qty)} 件）'

        # 取得關聯的訂單明細
        orders = []
        try:
            rows = db.session.query(OrderItem, Order).join(Order, OrderItem.order_id == Order.id).filter(OrderItem.item_id == it.id).all()
            for oi, o in rows:
                buyer_name = None
                if o.user_id:
                    u = User.query.get(o.user_id)
                    if u:
                        buyer_name = u.name
                # 取買家姓氏（簡單取 name 的第一個字/token）
                buyer_surname = None
                buyer_id = None
                if o.user_id and u:
                    buyer_id = u.id
                    # 嘗試用空白分割取第一個 token，若無再取第一個字
                    try:
                        parts = u.name.split()
                        if len(parts) > 0 and parts[0]:
                            buyer_surname = parts[0][0]
                        else:
                            buyer_surname = u.name[0]
                    except Exception:
                        buyer_surname = (u.name or '')[:1]

                orders.append({
                    'order_id': o.id,
                    'buyer_name': buyer_name or '匿名',
                    'buyer_surname': buyer_surname or '匿名',
                    'buyer_id': buyer_id,
                    'order_time': o.created_at,
                    'quantity': int(oi.quantity),
                    'contact_phone': getattr(o, 'contact_phone', None),
                    'delivery_method': getattr(o, 'delivery_method', None),
                    'pickup_location': getattr(o, 'pickup_location', None),
                    'shipping_address': getattr(o, 'shipping_address', None)
                })
        except Exception:
            orders = []

        items_info.append({
            'item': it,
            'sales_qty': int(total_qty),
            'sales_status': sales_status,
            'is_active': bool(it.is_active),
            'orders': orders
        })

    return render_template('seller_center.html', items_info=items_info)


@app.route('/item/<int:item_id>/mark_completed', methods=['POST'])
def mark_item_completed(item_id):
    """賣家手動標示該商品已完成交貨與收款：此操作會把商品下架並將庫存設為 0（作為標記）。"""
    item = Item.query.get_or_404(item_id)
    current_user_id = session.get('user_id')
    if not current_user_id or int(current_user_id) != int(item.seller_id):
        return "未授權", 403

    try:
        item.is_active = False
        item.quantity = 0
        db.session.commit()
    except Exception:
        db.session.rollback()
        session['buy_error'] = '標示完成失敗，請稍後再試。'
        return redirect(url_for('seller_center'))

    return redirect(url_for('seller_center'))


@app.route('/order/<int:order_id>')
def order_detail(order_id):
    o = Order.query.get_or_404(order_id)
    # 取買家
    buyer_name = None
    buyer_id = None
    buyer_surname = None
    if o.user_id:
        u = User.query.get(o.user_id)
        if u:
            buyer_name = u.name
            buyer_id = u.id
            try:
                parts = u.name.split()
                buyer_surname = parts[0][0] if parts and parts[0] else u.name[0]
            except Exception:
                buyer_surname = (u.name or '')[:1]

    # 取出 order items 與對應 item 資料
    items = []
    try:
        oitems = OrderItem.query.filter_by(order_id=order_id).all()
        for oi in oitems:
            itm = Item.query.get(oi.item_id)
            items.append((oi, itm))
    except Exception:
        items = []

    return render_template('order_detail.html', order=o, buyer_name=buyer_name, buyer_id=buyer_id, buyer_surname=buyer_surname, items=items)

# -----------------------------
# 問答區（買家/賣家）route
# -----------------------------
@app.route("/qa/<int:item_id>")
def qa(item_id):
    try:
        item = Item.query.get_or_404(item_id)
        # 確保 messages table 已建立
        messages = Message.query.filter_by(item_id=item_id).order_by(Message.timestamp).all()
        # 製作一份帶發言者名稱的訊息集合供模板使用
        messages_out = []
        for m in messages:
            user_name = None
            if m.user_id:
                u = User.query.get(m.user_id)
                if u:
                    user_name = u.name
            display = None
            if user_name:
                display = user_name
            else:
                # 若沒有 user_name，使用 sender（buyer/seller）的通用名稱
                # 對未登入的買家使用 '匿名消費者'，避免出現「消費者消費者」重複字樣
                display = '賣家' if m.sender == 'seller' else '匿名消費者'
            messages_out.append({'id': m.id, 'sender': m.sender, 'msg': m.msg, 'user_id': m.user_id, 'display_name': display, 'timestamp': m.timestamp})

        current_user_id = session.get('user_id')
        return render_template("qa.html", item=item, messages=messages_out, current_user_id=current_user_id)
    except Exception as e:
        return f"<h2>資料載入錯誤</h2><p>{e}</p>"

# -----------------------------
# SocketIO：問答區
# -----------------------------
@socketio.on('join')
def handle_join(data):
    room = data['room']
    join_room(room)

@socketio.on('send_message')
def handle_send_message(data):
    room = data.get('room')
    msg = data.get('msg')
    item_id = data.get('item_id')

    try:
        user_id = session.get('user_id')
        # 根據 session 與 item 賣家關係推斷身分
        item = Item.query.get(item_id)
        if item and item.seller_id and user_id and int(item.seller_id) == int(user_id):
            sender = 'seller'
        else:
            sender = 'buyer'

        # 如果 sender 被判定為 seller，但 item 沒有 seller_id 或不符，拒絕
        if sender == 'seller' and (not item or not item.seller_id or int(item.seller_id) != int(user_id)):
            emit('message', {'sender': 'system', 'msg': '未授權的賣家發言'}, room=room)
            return

        # 建立 display_name
        display_name = None
        if user_id:
            u = User.query.get(user_id)
            if u:
                display_name = u.name
        if not display_name:
            # 未登入的買家使用 '匿名消費者' 作為顯示名稱，避免與角色標籤重複
            display_name = '賣家' if sender == 'seller' else '匿名消費者'

        # 存資料庫（紀錄 user_id，如果有）
        message = Message(item_id=item_id, sender=sender, msg=msg, user_id=user_id)
        db.session.add(message)
        db.session.commit()
    except Exception as e:
        print("儲存訊息失敗:", e)
        emit('message', {'sender': 'system', 'msg': f'伺服器儲存失敗: {e}'}, room=room)
        return

    # 廣播給房間，包含 display_name 與 user_id
    emit('message', {'sender': sender, 'msg': msg, 'user_id': user_id, 'display_name': display_name}, room=room)


# -----------------------------
# SocketIO：私訊會話
# -----------------------------
@socketio.on('join_conv')
def handle_join_conv(data):
    conv_id = data.get('conv_id')
    room = f'conv_{conv_id}'
    join_room(room)
    # 當使用者加入會話時，將屬於此會話、發給當前使用者的未讀訊息標成已讀，並通知房間
    try:
        # Prefer Flask-Login current_user id if available, fall back to session token
        try:
            uid = current_user.id if (current_user and getattr(current_user, 'is_authenticated', False)) else session.get('user_id')
        except Exception:
            uid = session.get('user_id')
        if uid:
            msgs = Message.query.filter_by(conversation_id=conv_id, is_read=False).filter(Message.user_id != uid).all()
            ids = []
            for m in msgs:
                m.is_read = True
                m.read_at = datetime.utcnow()
                ids.append(m.id)
                db.session.add(m)
            if ids:
                db.session.commit()
                emit('read_receipt', {'conv_id': conv_id, 'message_ids': ids}, room=room)
                # 推送給當前連線的 client 最新的未讀交談數（只回自己的 socket）
                try:
                    from models import Message, Conversation
                    new_cnt = db.session.query(func.count(func.distinct(Message.conversation_id))).join(Conversation, Message.conversation_id == Conversation.id).filter(
                        Message.is_read == False,
                        Message.user_id != uid,
                        ((Conversation.user1_id == uid) | (Conversation.user2_id == uid))
                    ).scalar() or 0
                    emit('unread_conversations', {'count': int(new_cnt)}, room=request.sid)
                except Exception as e:
                    print('emit unread count failed:', e)
    except Exception as e:
        db.session.rollback()
        print('join_conv mark read error:', e)


@socketio.on('conv_send')
def handle_conv_send(data):
    conv_id = data.get('conv_id')
    msg = data.get('msg')
    image_url = data.get('image_url')
    image_filename = data.get('image_filename')
    try:
        # Prefer Flask-Login current_user id when available
        try:
            uid = current_user.id if (current_user and getattr(current_user, 'is_authenticated', False)) else session.get('user_id')
        except Exception:
            uid = session.get('user_id')
        # 建立並儲存訊息
        # 儲存檔名於 DB（image_path），並廣播完整可存取的 image_url
        stored_name = None
        if image_filename:
            stored_name = image_filename
            image_url_to_broadcast = url_for('uploaded_file', filename=stored_name)
        else:
            # 支援舊的 image_url 欄位介面
            image_url_to_broadcast = image_url
            # 若只有 image_url，但是 /uploads/xxx，嘗試從 URL 擷取檔名並存進 DB
            try:
                if image_url and isinstance(image_url, str):
                    # 如果包含完整 host，摘出 path
                    from urllib.parse import urlparse
                    p = urlparse(image_url)
                    candidate = os.path.basename(p.path)
                    if candidate:
                        stored_name = candidate
            except Exception as e:
                print('extract filename from image_url failed:', e)
        print(f"handle_conv_send: conv={conv_id} uid={uid} stored_name={stored_name} image_url={image_url_to_broadcast}")
        m = Message(conversation_id=conv_id, sender='user', msg=msg or '', user_id=uid, image_path=stored_name)
        db.session.add(m)
        db.session.commit()
        # 廣播給房間
        room = f'conv_{conv_id}'
        display_name = None
        if uid:
            u = User.query.get(uid)
            if u:
                display_name = u.name
        if not display_name:
            display_name = '使用者'
        emit('conv_message', {'id': m.id, 'conv_id': conv_id, 'user_id': uid, 'display_name': display_name, 'msg': m.msg, 'image_url': image_url_to_broadcast, 'timestamp': m.timestamp.isoformat()}, room=room)
        # 回傳 ack 給發送端
        return {'status': 'ok', 'message_id': m.id}
    except Exception as e:
        db.session.rollback()
        print('conv_send error:', e)
        emit('conv_error', {'error': str(e)}, room=f'conv_{conv_id}')
        return {'status': 'error', 'error': str(e)}

# -----------------------------
# 啟動 Flask + SocketIO
# -----------------------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # 確保 Item 與 Message table 都建立
    # 當使用 eventlet 與 debug 模式時，reloader 可能會導致綁定衝突，關閉 reloader 可避免 WinError 10048
    socketio.run(app, debug=True, use_reloader=False)
