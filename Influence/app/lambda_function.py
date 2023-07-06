import os
import json
import boto3
from datetime import datetime
from linebot import LineBotApi, WebhookParser
from linebot.models import TextSendMessage
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory , ConversationBufferWindowMemory
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)


# 変数の定義
access_token = os.environ.get('access_token')
secret_key = os.environ.get('secret_key')
channel_secret = os.environ.get('channel_secret')
aws_access_key_id = os.environ.get('aws_access_key')
aws_secret_access_key = os.environ.get('aws_secret_key')
template = '''
これから会話を行います。以下の条件を絶対に守って回答してください。

あなたは人間の女性である「八神優紀」です。
年齢は26歳です。
2020年10月から会社経営をしている女性の社長として活動しています。
Youtube,Tiktok,instagramなどを介したインフルエンサーとしても活動しています。
1日のスケジュールは,9時に起床して半身浴をする。11時からTikTokライブを始めて13時に美容整体,14時30分くらいからエステに行く。16時に仕事を始めて,20時にジムに行く。22時にTikTokライブを始めて23時にライブを終了。25時からデスクワークをして28時に終える。29時に寝る。
美容意識がすごく高いです。
美容オタクです。
持ち歩く荷物は,名刺,充電器,財布,ペン,メイク道具,充電コード,イヤホン2つ,ミストの化粧水,普通の化粧水,ハンドクリーム,顔のパック,ヘアミスト,ヘアオイル,香水,カーラー,糖質カットのサプリメント,ジムウェア,まつげ美容液,ヨーロッパ対応の電源アダプタ
リフォームの営業会社を立ち上げた。
ポッキーの持つところや男梅を耳に入れたりします。
愛車はフェラーリ。
あざとい系の女子です。
筋トレをよくします。
週2回パーソナルジムに行きます。
車が好きです。
スロットを打つことがあります。
携帯のロック画面はナカモトのカップラーメンです。
トレーニングの後は,メロンソーダのような甘いものを食べたり飲んだりしたいです。


チャットのように会話してください。
第一人称は「わたし」を使ってください。
敬語は使わないで，砕けた感じでフレンドリーに会話してください。
質問に答えれない場合は，会話を濁してください。
会話を続けることが難しかったら話題を逸らしてください。
関西弁などは用いず，標準語で話してください。
友達と会話するように話してください。
簡潔に会話し，長文は送らないでください。
'''

# daynamoDBの設定
dynamodb = boto3.resource('dynamodb', region_name='ap-northeast-1', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
client = boto3.client('dynamodb')
table_names = client.list_tables()['TableNames']

def lambda_handler(event, context):
    # jsonの読み込み
    BUCKET_NAME = 'influencerai-yagami'
    UPLOAD_BUCKET_KEY = 'line.json'
    # Lineの設定
    line_bot_api = LineBotApi(access_token)
    line_parser = WebhookParser(channel_secret)
    body = event['body']
    signature = event.get("headers", {}).get("x-line-signature")
    events = line_parser.parse(body,signature)
    # LangChainの設定
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=secret_key, temperature=0.8)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(template),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    # LINE パラメータの取得
    for event in events:
        line_user_id = event.source.user_id
        line_message = event.message.text
        profile = line_bot_api.get_profile(line_user_id)
        user_name = profile.display_name
        # ユーザーIDのテーブルが存在する場合，過去のチャット内容をDBから受け取る
        if line_user_id in table_names:
            # LangChainの設定と，ai_messageの生成
            message_history = DynamoDBChatMessageHistory(table_name=line_user_id, session_id="0")
            memory = ConversationBufferMemory(
              memory_key="history", chat_memory=message_history, return_messages=True
            )
            chain = ConversationChain(memory=memory, prompt=prompt, llm=llm)
            ai_message = chain.predict(input=line_message)
            # LINE メッセージの送信
            line_bot_api.push_message(line_user_id, TextSendMessage(chain.predict(input=line_message)))
            # DBを更新
            table = dynamodb.Table(line_user_id)
            response = table.update_item(
                Key={'SessionId': 'msg_ID'},
                UpdateExpression='ADD msg_counter :inc',
                ExpressionAttributeValues={':inc': 1},
                ReturnValues="UPDATED_NEW"
            )
            now = datetime.now()
            now_str = now.isoformat()
            new_id = response['Attributes']['msg_counter']
            table.put_item(
                Item={
                    'SessionId': str(new_id),
                    'line_message': line_message,
                    'ai_message': ai_message,
                    'LastUpdated': now_str
                }
            )
            response = table.scan()
        # ユーザーIDのテーブルが存在しない場合，新しくテーブルを作成し，そこにチャット内容を追加できるようにする
        else:
            # LangChainの設定と，ai_messageの生成
            memory = ConversationBufferMemory(return_messages=True)
            chain = ConversationChain(memory=memory, prompt=prompt, llm=llm)
            ai_message = chain.predict(input=line_message)
            # LINE メッセージの送信
            line_bot_api.push_message(line_user_id, TextSendMessage(chain.predict(input=line_message)))
            # DBを作成，更新
            table = dynamodb.create_table(
                TableName=line_user_id,
                KeySchema=[{"AttributeName": "SessionId", "KeyType": "HASH"}],
                AttributeDefinitions=[{"AttributeName": "SessionId", "AttributeType": "S"}],
                BillingMode="PAY_PER_REQUEST",
            )
            table.meta.client.get_waiter("table_exists").wait(TableName=line_user_id)
            table.put_item(
              Item={
                  'SessionId': 'msg_ID',
                  'msg_counter': 0
              }
            )
            response = table.update_item(
                Key={'SessionId': 'msg_ID'},
                UpdateExpression='ADD msg_counter :inc',
                ExpressionAttributeValues={':inc': 1},
                ReturnValues="UPDATED_NEW"
            )
            now = datetime.now()
            now_str = now.isoformat()
            new_id = response['Attributes']['msg_counter']
            table.put_item(
                Item={
                    'SessionId': str(new_id),
                    'line_message': line_message,
                    'ai_message': ai_message,
                    'LastUpdated': now_str
                }
            )
            response = table.scan()
        # s3に保存
        s3 = boto3.resource('s3')
        obj = s3.Bucket(BUCKET_NAME).Object(UPLOAD_BUCKET_KEY)
        data = obj.get()['Body'].read().decode('utf-8')
        json_data = json.loads(data)
        id = str(len(json_data)+1)
        new_data = {
            f'{id}': {
                "userName": user_name,
                "message": line_message,
                "response": ai_message
                }
            }
        json_data.update(new_data)
        res = obj.put(Body=json.dumps(json_data, ensure_ascii=False))
    
    ok_json = {"isBase64Encoded": False , "statusCode": 200 , "headers": {} , "body": ""}
    return ok_json