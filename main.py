import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import japanize_matplotlib
import json

model = SentenceTransformer('all-MiniLM-L6-v2')

# sentences = [
#     "掘り下げることが好きすぎて\n1行の好意的な感想より\n10行の批判的な感想のが喜ぶオタクです。\nINTP　△594　Ne-LII　言語優位　9種(1種6種)　　\n\n\n\n🟠好きな映画\nバタフライエフェクト/羊たちの沈黙/ヘレディタリー継承/セッション\nハムナプトラ/天使にラブソングを/ショーシャンクの空に/バーフバリ\nスターウォーズep5/ゼログラビティ/DUNE/オデッセイ/メッセージ\n\n🟠好きなアニメ\n進撃の巨人/Steins;Gate/新世紀エヴァンゲリオン/宝石の国\nモンタナジョー ンズ/四畳半神話大系/トムとジェリー/みるタイツ\nオトナ帝国の逆襲/カリオストロの城/パーフェクトブルー\n\n🟡好きなゲームジャンル\nシナリオゲー/テキストノベル/ADV/フリゲ/インディーゲー\n\n🟡好きなゲーム作品\nSIREN/パワポケ(怪奇ハタ人間篇)/ダンガンロンパ/Chaos;Child/アマガミ\nUndertale/ドキ文/Inscryption/パラノマサイト/Chants of Sennaar/東方project\nデンシャ/寄生ジョーカー/神林家殺人事件/クラッシュバンディクー\n\n🟢好きな作家\n秋山瑞人(イ リヤの空、UFOの夏/猫の地球儀)\n星新一(ひとつの装置/隊員たち)\n飲茶(史上最強の哲学入門/正義の教室)\n\n🟢好きな書籍\n古典(源氏物語)　　SF小説(星を継ぐもの/玩具修理者)\nSFラノベ(タイムリープ〜あしたはきのう〜/紫色のクオリ ア)\n人文書(歌うネアンデルタール/サピエンス全史/訂正可能性の哲学)\n\n🔵好きな音楽ジャンル・特徴\nゲーム音楽/chiptune/民族音楽/citypop/懐メロ/歌謡/vaporwave\nsignalwave/吹奏楽/ゴスペル/合唱/アカペラ/8分の6拍子/ドラムの演奏\n\n🔵好きな作曲家・歌手\nsasakure.UK/ピノキオピー/腹話/やながみゆき/ピーナッツくん\nZUN/TobyFox/平沢進/P-MODEL/MichaelJackson/Pentatonix\n\n\n\n🟣興味のある学問・教養ジャンル🟣\n言語学/日本語学/文芸評論/SF/科学史/量 子論/宇宙論/天文\n哲学史/東洋思想/古代史/先史/考古学/文化人類学/神話/宗教/民俗文化\n地理学/地政学/進化生物学/分類学/進化心理学/精神医学/音楽理論\n浅く広く怪しげな知識/スケールの大き(過ぎて役に立たな)い話\n\n⚪そのほか興味ある/好きなコンテンツ⚪\n性格診断・認知特性(視覚/言語等)・体癖・創作神話・言語創作\n天原(漫画)・わたモテ(7~14巻)・TRPG・マダミス・ボドゲ等\n\n俺新訳ダンガンロンパ(二次創作,長編小説)\nオービタルラビット(東方二次創作,MMD映画)\n超幻想郷級のダンガンロンパ(クロスオーバー,動画シリーズ)\n\n\n\n🔴見てるYouTube/ニコニコのチャンネル・投稿者🔴\n解説　　　　 　河江肖剰の古代エジプト/守鍬刈雄のお暇なら映画でも\n　　　　　　　 山田玲司のヤングサ ンデー/山田五郎オトナの教養講座\n　　　　　　　 社會部部長/音楽ガチ分析ch/岡田斗司夫\nゆっくり解説 　世界の奇書/進化生物学ch/地理の雑学/3分でわかる宗教解説\n　　　　　　　 ゲーム夜話/ぴよぴーよ速報/るーい科学/デルタ科学/9割雑学\n\nゲーム実況　　稲葉百万鉄/ゲームさんぽ/名越康文/ガッkoya\n動画勢Vtuber　ぽんぽこ/ピーナッツくん/ヘアピンまみれ/MZM/銀河アリス\n\nエンタメ　ホッカイロレン/Quizknock/バキ童/オモコロ/Vクエ\n二次創作　KamS(東方手書き)/2151(そばかすMMD)/tehcno-m@ster(平沢AMV)\n\n言語　すきえんてぃあ(minerva scientia)/ゆる言語学ラジオ/国立国語研究所\n　　　坂本小見山/いのほた言語学/け゚とま/言語の部屋/ことラボ/日曜言語学\n文芸　又吉直樹(インスタ   ントフィクションシリーズ)/夏井いつき俳句ch\n\n時事　東浩紀(ゲンロン)/中田敦彦/トーマスガジェマガ/WeeklyOchiai等\n国際　Dogen/プク太/DALT\n",
#     "プログラミングは私の趣味の一つです。",
#     "今日は天気がいいですね。",
#     "言語をながめるのが好きですが習得は苦手です。飽き性でいろいろなことをぐるぐるやりがちです。\n他人との交流も求めますが、気を遣いすぎがちなのでひとりの時間が必須です。MBTIはIxxPみたいな感じです（S–N、T–Fがそ れぞれ同程度っぽい）。\n\n【好きなもの】\n音楽\nわりと最近～いまの\n▸ BUMP OF CHICKEN / amazarashi / Vaundy / ヨルシカ / セラニポージ / スピッツ / Yellow Magic Orchestra / Kraftwerk / 相対性理論 / ニコライ ・カプースチン / The Dave Brubeck Quartet\n歴史上の\n▸ ベートーヴェン / ドビュッシー / サティ\n\nゲーム\n任天堂\n▸ ゼルダシリーズ、ピクミンシリーズ、パネルでポン\nほか\n▸ Detroit: Become Human / Ghost of Tsushima / Factorio / Simcity シリーズ (3000 が一番好きだった) / OMORI / FTL: Faster Than Light / ファミ レスを享受せよ / Somi（リーガルダンジョン / 未解決事件は終わらせないといけないから）/ Chants of Sennaar\n\n小説・漫画\n九井諒子（「ダンジョン飯」ほか）/ Kazuo Ishiguro (Klara and the Sun, etc.) / Ken Liu (The Paper Menagerie, etc)\n\n学術\n言語学\n▸ 音韻・音声＞形態/統語＞＞意味・語用・談話（興味順）\n▸ 古典語もわかりたい\nほか\n▸ 文字・書体・タイポグラフィー / Unicode / フォント技術 (OpenType)\n\nYouTubeチャンネル\nゲーム関係\n▸ ナポリの男たち（特に ジャック・オ・蘭たん / すぎる）/ P-P / 稲葉百万鉄 / ゲームさんぽ (株式会社よそ見)\n VTuber \n▸ にじさんじ（特に フレン・E・ルスタリオ、アンジュ・カトリーナ）\n音楽関係\n▸ 8-bit Music Theory / Charles Cornell / David Bennett Piano\nその他\n▸ QuizKnock（河村さん・とむさん推し） / Dr Geoff Lindsey（音声学者）\n\nその他\nラーメンズ / 名越康文 / シナモン（サンリオ）\n\n同時視聴可能なサブスク\nAmazon Prime Video\n"
# ]

sentences = []


json_open = open('test.json', 'r',encoding="utf-8_sig")
json_load = json.load(json_open)

for p in json_load:
    print(p["name"])
    sentences.append(p["description"])


sentence_embeddings = model.encode(sentences)

similarity_matrix = cosine_similarity(sentence_embeddings)

tsne = TSNE(n_components=2, random_state=42, perplexity=2)
embeddings_2d = tsne.fit_transform(sentence_embeddings)

plt.figure(figsize=(10, 6))
for i, sentence in enumerate(sentences):
    name = json_load[i]["name"]
    plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1])
    plt.text(embeddings_2d[i, 0] + 0.1, embeddings_2d[i, 1], name, fontsize=12)

plt.title("t-SNE visualization of sentence embeddings")
plt.show()

# plt.figure(figsize=(8, 6))
# sns.heatmap(similarity_matrix, annot=True, xticklabels=sentences, yticklabels=sentences, cmap='coolwarm')
# plt.title("Cosine Similarity Matrix")
# plt.show()