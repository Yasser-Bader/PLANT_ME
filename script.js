let model;
let class_indices;
let fileUpload = document.getElementById('uploadImage')
let img = document.getElementById('image')
let boxResult = document.querySelector('.box-result')
let confidence = document.querySelector('.confidence')
let pconf = document.querySelector('.box-result p')
/*
$("textarea").each(function () {
    this.setAttribute("style", "height:" + (this.scrollHeight) + "px;overflow-y:hidden;");
  }).on("input", function () {
    this.style.height = 0;
    this.style.height = (this.scrollHeight) + "px";
  });*/
        
        let progressBar = 
            new ProgressBar.Circle('#progress', {
            color: 'limegreen',
            strokeWidth: 10,
            duration: 2000, // milliseconds
            easing: 'easeInOut'
        });

        async function fetchData(){
            let response = await fetch('./class_indices.json');
            let data = await response.json();
            data = JSON.stringify(data);
            data = JSON.parse(data);
            return data;
        }

         // here the data will be return.
        

        // Initialize/Load model
        async function initialize() {
            let status = document.querySelector('.init_status')
            status.innerHTML = 'جار التحميل .... <span class="fa fa-spinner fa-spin"></span>'
            model = await tf.loadLayersModel('./tensorflowjs-model/model.json');
            status.innerHTML = 'تم تحميل النموذج بنجاح  <span class="fa fa-check"></span>'
        }

        async function predict() {
            // Function for invoking prediction
            let img = document.getElementById('image')
            let offset = tf.scalar(255)
            let tensorImg =   tf.browser.fromPixels(img).resizeNearestNeighbor([224,224]).toFloat().expandDims();
            let tensorImg_scaled = tensorImg.div(offset)
            prediction = await model.predict(tensorImg_scaled).data();
           
            fetchData().then((data)=> 
                {
                    predicted_class = tf.argMax(prediction)
                    
                    class_idx = Array.from(predicted_class.dataSync())[0]
                    document.querySelector('.pred_class').innerHTML = data[class_idx]
                    document.querySelector('.inner').innerHTML = `متأكد ${parseFloat(prediction[class_idx]*100).toFixed(2)}%`
                    console.log(data)
                    console.log(data[class_idx])
                    console.log(prediction)

                    progressBar.animate(prediction[class_idx]-0.005); // percent

                    pconf.style.display = 'block'

                    confidence.innerHTML = Math.round(prediction[class_idx]*100)
                    if(data[class_idx] == "Potato___Early_blight")
                    {

                        document.querySelector('.read').innerHTML =`نبات البطاطس مصاب باللفحة المبكره
                        الأعراض
                        1-تظهر أولى أعراض المرض على هيئة بقع داكنة مع نمو متحد المركز وهالات صفراء على الأوراق.
                        2-متبوع بجفاف الأوراق وتساقطها .
                        3-الثمار قد تبدأ بالتعفن ثم تسقط في نهاية الأمر.
                        تحدث أعراض اللفحة المبكرة على أوراق الشجر القديمة والسيقان ، والثمار . تظهر بقع ذات لون رمادي مائل إلى البني على أوراق الشجر ، وتنمو تدريجياً بطريقة مركزية حول مركز واضح- مما يشكل المظهر المميز الذي يشبه " هدف الرماية " . تكون هذه البقع محاطة بهالة صفراء زاهية ومع تقدم المرض ، قد تتحول الأوراق بأكملها إلى اللون الأصفر الناجم عن نقص الكلوروفيل وتتساقط ، مما يؤدي إلى تساقط الأوراق بشكل ملحوظ . وعندما تموت الأوراق وتسقط ، تصبح الثمار أكثر عرضة لحروق الشمس . ويظهر نفس النوع من البقع ذات المركز الواضح على السيقان والفواكه. ويبدأ تعفن الفواكه وقد تتساقط في النهاية
                        اجراءت وقائية
                        يجب عليك استخدام بذور معتمدة أو خالية من مسببات الأمراض. 
                        والبحث عن الأصناف المقاومة للمرض.
                        يجب زراعة البذور أو الشتلات على مهاد مرتفع ومحروث لتحسين الصرف.
                        وتأكد من توجيه الصفوف في اتجاه الرياح الرئيسية وتجنب المناطق المظللة.
                        وحافظ على مساحة مناسبة بين النباتات للسماح للمظلة الخضرية أن تجف بسرعة بعد هطول الأمطار أو الري.
                        كما يجب وضع المهاد على التربة لمنع النباتات من لمس التربة
                        وكذلك مراقبة الحقول بحثا عن علامات المرض ، خاصة أثناء الطقس الرطب.
                        وإزالة الأوراق السفلية القريبة جداً من التربة.
                        وإزالة الأوراق التي تمت ملاحظة الأعراض عليها وتدميرها.
                        وكذلك الحفاظ على النباتات قوية ومخضرة بإضافة ما يكفي من المغذيات.
                        وكذلك استخدام العصي الخشبية للحفاظ على النباتات منتصبة ومستقيمة.
                        استخدام نظام الري بالتنقيط للحد من بلل الأوراق.
                        وقم بالري في الصباح حتى تحافظ على النباتات جافة خلال النهار.
                        وقم أيضاً بمكافحة الحشائش القابلة للإصابة في الحقل ومن حوله.
                        أما بعد الحصاد ، فقم بإزالة المخلفات النباتية وحرقها ( ولا تقم بتحويلها إلى سماد بطريقة الكومبوست ).
                        بدلاً من ذلك ، يمكنك استخدام الحرث العميق لدفن المخلفات عميقاً في التربة ( أكثر من 45 سم ).
                        وتخطيط وتنفيذ تناوب المحاصيل لمدة 2 أو 3 سنوات مع المحاصيل غير المضيفة أو المعرضة للإصابة.
                        وقم بتخزين الثمار في درجات حرارة باردة وفي مواقع جيدة التهوية.
                        المكافحة الكيميائية
                        1-مبيدات الفطرية proineb 70.0 WP
                        2-مبيدات البكتريا copper oxychloride 50.0 WP
                        3-مبيدات الفطريات mancozeb 75.0 WP
                        سبب المرض
                        تنجم الأعراض عن فطر ألترناريا سولاني ، وهي فطريات تعيش على بقايا المحاصيل المصابة في التربة أو على العوائل البديلة . كما قد تكون البذور المشتراة أو الشتلات ملوثة بالفعل . وغالبا ما تصاب الأوراق السفلي عند ملامستها للتربة الملوثة . وتعد درجات الحرارة الدافئة ( 29-24 درجة مئوية ) والرطوبة العالية ( 90 % ) ظروفاً مواتية لتطور المرض . ولذلك تعمل الفترات الرطبة الطويلة ( أو الطقس الرطب / الجاف بالتناوب ) على تحسين إنتاج الأبواغ الفطرية ، والتي قد تنتشر عن طريق الرياح أو تناثر قطرات المطر أو الري العلوي . تكون الثمار التي يتم حصادها ولونها أخضر أو في الظروف الرطبة معرضة بشكل خاص للعدوى . وغالباً ما ينتشر المرض بشدة بعد فترة من الأمطار الغزيرة ، ويعد من الأمراض المدمرة خصوصاً في المناطق الاستوائية وشبه الاستوائية .`

                        document.querySelector('.print').innerHTML =`<h2>نبات البطاطس مصاب باللفحة المبكرة</h2>
                        <h2>🍂 الأعراض</h2>
                        1-تظهر أولى أعراض المرض على هيئة بقع داكنة مع نمو متحد المركز وهالات صفراء على الأوراق.<br>
                        2-متبوع بجفاف الأوراق وتساقطها .<br>
                        3-الثمار قد تبدأ بالتعفن ثم تسقط في نهاية الأمر.<br>
                        تحدث أعراض اللفحة المبكرة على أوراق الشجر القديمة والسيقان ، والثمار . تظهر بقع ذات لون رمادي مائل إلى البني على أوراق الشجر ، وتنمو تدريجياً بطريقة مركزية حول مركز واضح- مما يشكل المظهر المميز الذي يشبه " هدف الرماية " . تكون هذه البقع محاطة بهالة صفراء زاهية ومع تقدم المرض ، قد تتحول الأوراق بأكملها إلى اللون الأصفر الناجم عن نقص الكلوروفيل وتتساقط ، مما يؤدي إلى تساقط الأوراق بشكل ملحوظ . وعندما تموت الأوراق وتسقط ، تصبح الثمار أكثر عرضة لحروق الشمس . ويظهر نفس النوع من البقع ذات المركز الواضح على السيقان والفواكه. ويبدأ تعفن الفواكه وقد تتساقط في النهاية
                        <br>
                        <h2>اجراءات وقائية</h2>
                        يجب عليك استخدام بذور معتمدة أو خالية من مسببات الأمراض. <br>
                        والبحث عن الأصناف المقاومة للمرض.<br>
                        يجب زراعة البذور أو الشتلات على مهاد مرتفع ومحروث لتحسين الصرف.<br>
                        وتأكد من توجيه الصفوف في اتجاه الرياح الرئيسية وتجنب المناطق المظللة.<br>
                        وحافظ على مساحة مناسبة بين النباتات للسماح للمظلة الخضرية أن تجف بسرعة بعد هطول الأمطار أو الري.<br>
                        كما يجب وضع المهاد على التربة لمنع النباتات من لمس التربة.<br>
                        وكذلك مراقبة الحقول بحثا عن علامات المرض ، خاصة أثناء الطقس الرطب.<br>
                        وإزالة الأوراق السفلية القريبة جداً من التربة.<br>
                        وإزالة الأوراق التي تمت ملاحظة الأعراض عليها وتدميرها.<br>
                        وكذلك الحفاظ على النباتات قوية ومخضرة بإضافة ما يكفي من المغذيات.<br>
                        وكذلك استخدام العصي الخشبية للحفاظ على النباتات منتصبة ومستقيمة.<br>
                        استخدام نظام الري بالتنقيط للحد من بلل الأوراق.<br>
                        وقم بالري في الصباح حتى تحافظ على النباتات جافة خلال النهار.<br>
                        وقم أيضاً بمكافحة الحشائش القابلة للإصابة في الحقل ومن حوله.<br>
                        أما بعد الحصاد ، فقم بإزالة المخلفات النباتية وحرقها ( ولا تقم بتحويلها إلى سماد بطريقة الكومبوست ).<br>
                        بدلاً من ذلك ، يمكنك استخدام الحرث العميق لدفن المخلفات عميقاً في التربة ( أكثر من 45 سم ).<br>
                        وتخطيط وتنفيذ تناوب المحاصيل لمدة 2 أو 3 سنوات مع المحاصيل غير المضيفة أو المعرضة للإصابة.<br>
                        وقم بتخزين الثمار في درجات حرارة باردة وفي مواقع جيدة التهوية.<br>

                        <h2>⚗ المكافحة الكيميائية</h2>
                        1-مبيدات الفطرية proineb 70.0 WP<br>
                        2-مبيدات البكتريا copper oxychloride 50.0 WP<br>
                        3-مبيدات الفطريات mancozeb 75.0 WP<br>

                        <h2> سبب المرض</h2>
                        تنجم الأعراض عن فطر ألترناريا سولاني ، وهي فطريات تعيش على بقايا المحاصيل المصابة في التربة أو على العوائل البديلة . كما قد تكون البذور المشتراة أو الشتلات ملوثة بالفعل . وغالبا ما تصاب الأوراق السفلي عند ملامستها للتربة الملوثة . وتعد درجات الحرارة الدافئة ( 29-24 درجة مئوية ) والرطوبة العالية ( 90 % ) ظروفاً مواتية لتطور المرض . ولذلك تعمل الفترات الرطبة الطويلة ( أو الطقس الرطب / الجاف بالتناوب ) على تحسين إنتاج الأبواغ الفطرية ، والتي قد تنتشر عن طريق الرياح أو تناثر قطرات المطر أو الري العلوي . تكون الثمار التي يتم حصادها ولونها أخضر أو في الظروف الرطبة معرضة بشكل خاص للعدوى . وغالباً ما ينتشر المرض بشدة بعد فترة من الأمطار الغزيرة ، ويعد من الأمراض المدمرة خصوصاً في المناطق الاستوائية وشبه الاستوائية .
                        `

                    }if (data[class_idx] == "Pepper,_bell___Bacterial_spot" ) {
                        document.querySelector('.print').innerHTML =`<h2>نبات الفلفل مصاب ببقعة البكتريا</h2>
                        <h2>🍂 الأعراض</h2>
                        1-يظهر المرض علي شكل بقع صغيره صفراء مخضرة تظهر علي الاوراق حديثه النمو، اما الاوراق القديمه،  تظهر الافات علي  هيئة بقع داكنة مبللة بالماء ذات هالات صفراء. 
                        <br>
                        2-يحدث تشوه الاوراق و التفافها. 
                        <br>
                        3-وكذلك ظهور بقع مبللة علي الثمار والتي تصبح خشنة ولها قشرة بنيه.  
                        <br>
تظهر الاعراض الاولية علي شكل تقرحات صغيره صفراء مخضرة علي الاوراق حديثه النمو، التي غالبا ما تكون مشوهة و ملتفة.  اما علي الاوراق القديمه، فتظهر تقرحات زاويه ذات لون اخضر داكن ومظهر لزج. وغالبا ما تكون محاطة بهالات صفراء.  تكثر البقع علي حواف الاوراق وأطرافها عند تقدم المرض. في نهاية المطاف، تصبح البقع وكانها طلقات ناريه،  لان مركز البقعه يجف و ينفصل عن باقي نسيج الورقة. تبدا البقع في الظهور مبللة بالماء بلون اخضر باهت ذات مظهر لزج (يصل قطرها حتي 0.5سم) ومع تقدم المرض،  تتعفن البقع ويصبح لونها بني ذات قشرة.
                        <br>
                        <hr>
                        <h2>اجراءات وقائية</h2>
                        1- يجب عليك استخدام بذور خاليه من الامراض من مصادر موثوقه. 
                        <br>
                        2- واستخدام أصناف مقاومة إذا كانت متاحه في منطقتك. 
                        <br>
                        3- راقب الحقول بانتظام بحثا عن علامات المرض. 
                        <br>
                        4- قم بمراقبة الشتلات و إزالة أي شتلة أو نبات يظهر عليها التبقع البكتيري في الاوراق والتخلص منها بالحرق. 
                        <br>
                        5- يجب عليك التخلص من الحشائش في الحقل ومن حوله. 
                        <br>
                        6- استخدام التغطية البلاستيكية للتربة، والتي بدورها تمنع انتقال البكتيريا من التربة الي النبات.  
                        <br>
                        7- نظف المعدات والادوات الزراعيه بعد الاستخدام عند العمل بين حقول مختلفه. 
                        <br>
                        8- تجنب الري الرأسي والعمل في الحقل عندما تكون الاوراق مبللة. 
                        <br>
                        9- أحرث مخلفات النباتات عميقا في التربه بعد الحصاد او تخلص منها بعد الحصاد وأحرقها. 
                        <br>
                        10- ينصح  باتباع التناوب الزراعي مع محاصيل غير مضيفة من 2-3سنوات
                        <br>
                        <hr>
                        <h2>⚗ المكافحة الكيميائية</h2>
                        1-مبيدات الفطريات mancozeb 63.0 WP<br>
                        2-مبيدات البكتريا streomycin 90 SP<br>
                        <br>
                        <hr>
                        <h2> سبب المرض</h2>
                        يحدث مرض التبقع البكتيري في كل أنحاء العالم،  ويعتبر من الامراض الخطيرة علي الطماطم والفلفل التي تزرع في البيئات الدافئة الرطبة. يمكن لمسبب المرض البقاء علي لب البذور او العيش علي سطحها، واحيانا ما يبقي علي حشائش معينه تعتبر مضيفات بديلة. وينتشر فيما بعد عن طريق رذاذ الأمطار أو الري بالرشاشات العلوية. وعادة ما يدخل مسبب المرض الي أنسجة النباتات عبر مسامات الورقه والخدوش. وتعتبر درجة الحرارة المثلي لانتشار المرض من 25الي 30درجة مئوية. وفي حال حدوث الإصابة في المحصول، فإنه يكون من الصعب جدا السيطرة علي المرض او مكافحته، وقد يتسبب في فقدان كامل المحصول.
                        `

                        document.querySelector('.read').innerHTML =`نبات الفلفل مصاب ببقعة البكتريا
                        الأعراض
                        1-يظهر المرض علي شكل بقع صغيره صفراء مخضرة تظهر علي الاوراق حديثه النمو، اما الاوراق القديمه،  تظهر الافات علي  هيئة بقع داكنة مبللة بالماء ذات هالات صفراء. 
                        2-يحدث تشوه الاوراق و التفافها. 
                        3-وكذلك ظهور بقع مبللة علي الثمار والتي تصبح خشنة ولها قشرة بنيه.  
تظهر الاعراض الاولية علي شكل تقرحات صغيره صفراء مخضرة علي الاوراق حديثه النمو، التي غالبا ما تكون مشوهة و ملتفة.  اما علي الاوراق القديمه، فتظهر تقرحات زاويه ذات لون اخضر داكن ومظهر لزج. وغالبا ما تكون محاطة بهالات صفراء.  تكثر البقع علي حواف الاوراق وأطرافها عند تقدم المرض. في نهاية المطاف، تصبح البقع وكانها طلقات ناريه،  لان مركز البقعه يجف و ينفصل عن باقي نسيج الورقة. تبدا البقع في الظهور مبللة بالماء بلون اخضر باهت ذات مظهر لزج (يصل قطرها حتي 0.5سم) ومع تقدم المرض،  تتعفن البقع ويصبح لونها بني ذات قشرة.
                        اجراءات وقائية
                        1- يجب عليك استخدام بذور خاليه من الامراض من مصادر موثوقه. 
                        2- واستخدام أصناف مقاومة إذا كانت متاحه في منطقتك. 
                        3- راقب الحقول بانتظام بحثا عن علامات المرض. 
                        4- قم بمراقبة الشتلات و إزالة أي شتلة أو نبات يظهر عليها التبقع البكتيري في الاوراق والتخلص منها بالحرق. 
                        5- يجب عليك التخلص من الحشائش في الحقل ومن حوله. 
                        6- استخدام التغطية البلاستيكية للتربة، والتي بدورها تمنع انتقال البكتيريا من التربة الي النبات.  
                        7- نظف المعدات والادوات الزراعيه بعد الاستخدام عند العمل بين حقول مختلفه. 
                        8- تجنب الري الرأسي والعمل في الحقل عندما تكون الاوراق مبللة. 
                        9- أحرث مخلفات النباتات عميقا في التربه بعد الحصاد او تخلص منها بعد الحصاد وأحرقها. 
                        10- ينصح  باتباع التناوب الزراعي مع محاصيل غير مضيفة من 2-3سنوات
                        المكافحة الكيميائية
                        1-مبيدات الفطريات mancozeb 63.0 WP
                        2-مبيدات البكتريا streomycin 90 SP
                        سبب المرض
                        يحدث مرض التبقع البكتيري في كل أنحاء العالم،  ويعتبر من الامراض الخطيرة علي الطماطم والفلفل التي تزرع في البيئات الدافئة الرطبة. يمكن لمسبب المرض البقاء علي لب البذور او العيش علي سطحها، واحيانا ما يبقي علي حشائش معينه تعتبر مضيفات بديلة. وينتشر فيما بعد عن طريق رذاذ الأمطار أو الري بالرشاشات العلوية. وعادة ما يدخل مسبب المرض الي أنسجة النباتات عبر مسامات الورقه والخدوش. وتعتبر درجة الحرارة المثلي لانتشار المرض من 25الي 30درجة مئوية. وفي حال حدوث الإصابة في المحصول، فإنه يكون من الصعب جدا السيطرة علي المرض او مكافحته، وقد يتسبب في فقدان كامل المحصول.
                        `

                    } if (data[class_idx] == "Pepper,_bell___healthy") {
                        document.querySelector('.print').innerHTML =`<h2>نبات الفلفل غير مصاب (صحي)</h2>
                        <h2>نصائح للحفاظ علي صحة نباتك 🌿</h2>
                        يجب عليك شراء المواد الزراعية من مصادر معتمده. فحص التقاوي والشتلات بعناية قبل شرائها. السماح بمساحه تباعد كافية بين المحاصيل النباتية للسماح بالتهوية الجيدة. اختر الموقع (التربة والطقس) بعناية وتأكد من عدم زراعة أصناف حساسة . التسميد بخليط الأسمدة المناسب وإمداده بالمغذيات والتسميد المتوازن. وتجنب الإفراط في الري أو الإفراط في التسميد. لا تلمس النباتات الصحية بعد لمس النباتات المصابة. تجنب التغيرات الحادة في درجات الحرارة. الحفاظ علي عدد كبير من الأنواع مختلفة من النباتات حول الحقول. إذا كان العلاج ضد مرض معدي ويسهل انتشاره، استخدام منتجات محددة لا تؤثر علي الحشرات المفيدة. وتذكر إزالة الأوراق، الثمار أو الفروع المريضة في الوقت المناسب خلال موسم النمو. في الخريف، نظف بقايا النباتات من الحقل أو البستان وقم بحرقها.
                        `
                        document.querySelector('.read').innerHTML =`نبات الفلفل غير مصاب (صحي)
                        نصائح للحفاظ علي صحة نباتك 
                        يجب عليك شراء المواد الزراعية من مصادر معتمده. فحص التقاوي والشتلات بعناية قبل شرائها. السماح بمساحه تباعد كافية بين المحاصيل النباتية للسماح بالتهوية الجيدة. اختر الموقع (التربة والطقس) بعناية وتأكد من عدم زراعة أصناف حساسة . التسميد بخليط الأسمدة المناسب وإمداده بالمغذيات والتسميد المتوازن. وتجنب الإفراط في الري أو الإفراط في التسميد. لا تلمس النباتات الصحية بعد لمس النباتات المصابة. تجنب التغيرات الحادة في درجات الحرارة. الحفاظ علي عدد كبير من الأنواع مختلفة من النباتات حول الحقول. إذا كان العلاج ضد مرض معدي ويسهل انتشاره، استخدام منتجات محددة لا تؤثر علي الحشرات المفيدة. وتذكر إزالة الأوراق، الثمار أو الفروع المريضة في الوقت المناسب خلال موسم النمو. في الخريف، نظف بقايا النباتات من الحقل أو البستان وقم بحرقها.
                        `
                    } if (data[class_idx] == "Potato___healthy") {
                        document.querySelector('.print').innerHTML =`<h2>نبات البطاطس غير مصاب (صحي)</h2>
                        <h2>نصائح للحفاظ علي صحة نباتك 🌿</h2>
                        يجب عليك شراء المواد الزراعية من مصادر معتمده. فحص التقاوي والشتلات بعناية قبل شرائها. السماح بمساحه تباعد كافية بين المحاصيل النباتية للسماح بالتهوية الجيدة. اختر الموقع (التربة والطقس) بعناية وتأكد من عدم زراعة أصناف حساسة . التسميد بخليط الأسمدة المناسب وإمداده بالمغذيات والتسميد المتوازن. وتجنب الإفراط في الري أو الإفراط في التسميد. لا تلمس النباتات الصحية بعد لمس النباتات المصابة. تجنب التغيرات الحادة في درجات الحرارة. الحفاظ علي عدد كبير من الأنواع مختلفة من النباتات حول الحقول. إذا كان العلاج ضد مرض معدي ويسهل انتشاره، استخدام منتجات محددة لا تؤثر علي الحشرات المفيدة. وتذكر إزالة الأوراق، الثمار أو الفروع المريضة في الوقت المناسب خلال موسم النمو. في الخريف، نظف بقايا النباتات من الحقل أو البستان وقم بحرقها.
                        `
                        document.querySelector('.read').innerHTML =`نبات البطاطس غير مصاب (صحي)
                        نصائح للحفاظ علي صحة نباتك
                        يجب عليك شراء المواد الزراعية من مصادر معتمده. فحص التقاوي والشتلات بعناية قبل شرائها. السماح بمساحه تباعد كافية بين المحاصيل النباتية للسماح بالتهوية الجيدة. اختر الموقع (التربة والطقس) بعناية وتأكد من عدم زراعة أصناف حساسة . التسميد بخليط الأسمدة المناسب وإمداده بالمغذيات والتسميد المتوازن. وتجنب الإفراط في الري أو الإفراط في التسميد. لا تلمس النباتات الصحية بعد لمس النباتات المصابة. تجنب التغيرات الحادة في درجات الحرارة. الحفاظ علي عدد كبير من الأنواع مختلفة من النباتات حول الحقول. إذا كان العلاج ضد مرض معدي ويسهل انتشاره، استخدام منتجات محددة لا تؤثر علي الحشرات المفيدة. وتذكر إزالة الأوراق، الثمار أو الفروع المريضة في الوقت المناسب خلال موسم النمو. في الخريف، نظف بقايا النباتات من الحقل أو البستان وقم بحرقها.
                        `
                        
                    } if (data[class_idx] == "Potato___Late_blight") {
                        document.querySelector('.print').innerHTML =`<h2>نبات البطاطس مصاب باللفحة المتأخرة </h2>
                        <h2>🍂 الأعراض</h2>
                        1- تظهر اعراض المرض على هيئه بقع بنية داكنة تظهر على حواف الورقة وأطرافها.  <br>
                        2- ثم تظهر نمو فطري ابيض يغطي الجزء السفلي من الورقه.  <br>
                        3- مما يؤدي الي جفاف الاوراق وموتها.  <br>
                        4- كما وتظهر بقع زرقاء رمادية علي درنات البطاطس وتجعلها غير صالحة للاكل.  <br>
                        <br>
تظهر اعراض المرض علي هيئة بقع بنية داكنة تظهر علي حواف الورقة وأطرافها. أما في الظروف الجويه الرطبة، فتكتسب تلك البقع مظهرا لزجا و مبتلا.  كما ويمكن رؤية نمو فطري ابيض علي السطح السفلي للورقة. ومع تقدم المرض، تجف الورقة باكملها بعد تحولها للون البني ثم تموت. كما وتنتشر نفس البقع علي الساق و اعناق الاوراق. كما وتظهر بقع زرقاء رماديه علي قشور البطاطس ومن ثم يتحول اللب الي اللون البني مما يجعلها غير صالحة للاكل. تعفن الحقول المصابة يعطي رائحة مميزة.<br>
<hr>
                        <h2>اجراءات وقائية</h2>
                        1- عليك استخدام تقاوي سليمة و اصناف مقاومة للمرض. <br>
                        2- وتأكد ايضاً من وجود نظام تهويه وصرف جيدين. <br>
                        3- كما ويجب متابعه الزراعة والتخلص من النباتات المصابة وما حولها. <br>
                        4- ويجب أيضا إتباع التناوب الزراعي لمدة 2الي 3سنوات مع نباتات غير مضيفه للمرض. <br>
                        5- كما ويجب التخلص من المضيفات البديلة والنباتات الناتجة من نمو بقايا المحصول السابق من الحقل وما حوله. <br>
                        6- وعدم المبالغة في استخدام التسميد النيتروجينى. <br>
                        7- استخدام مقويات النباتات. <br>
                        8- كما ويجب تخزين البطاطس في درجات حرارة منخفضة و تهوية جيدة. <br>
                        9- والتخلص من بقايا المحصول السابق بعد الحصاد بدفنها عميقا تحت سطح التربة (اكثر من 60سم) او إطعامها الحيوانات.<br>
                        <hr>
                        <h2>⚗ المكافحة الكيميائية</h2>
                        1-مبيدات الفطريات hexaconazole 5.0 WP, captan 70.0 WP<br>
                        2-مبيدات الفطريات mancozeb 50.0 WS<br>
                        2-مبيدات الفطريات propineb 70.0 WP<br>
                        <br>
                        <hr>
                        <h2> سبب المرض</h2>
                        المسبب المرضي هو فطر إجباري التطفل. وهذا يعني ضرورة بقائة متصلا ببقايا النباتات و الدرنات او المضيفات البديلة. يمكن للفطر الدخول عبر الجروح والتشققات في سطح النبات. تستطيع الجراثيم النمو في درجات حرارة اعلي خلال فصل الربيع وتنتقل عبر الرياح و المياة. تشتد الاصابة في الاوقات ذات الليالي الباردة (اقل من 18مئوي)  والنهار الدافئ (بين 18و22مئوي) وظروف الرطوبة مثل الامطار والضباب (رطوبة نسبية 90%). في مثل تلك الظروف. يمكن ان تصبح آثار مرض اللفحة المتأخرة مدمرة.
                        `
                        document.querySelector('.read').innerHTML =`نبات البطاطس مصاب باللفحة المتأخرة
                         الأعراض
                        1- تظهر اعراض المرض على هيئه بقع بنية داكنة تظهر على حواف الورقة وأطرافها.  
                        2- ثم تظهر نمو فطري ابيض يغطي الجزء السفلي من الورقه.  
                        3- مما يؤدي الي جفاف الاوراق وموتها.  
                        4- كما وتظهر بقع زرقاء رمادية علي درنات البطاطس وتجعلها غير صالحة للاكل.  
                        
تظهر اعراض المرض علي هيئة بقع بنية داكنة تظهر علي حواف الورقة وأطرافها. أما في الظروف الجويه الرطبة، فتكتسب تلك البقع مظهرا لزجا و مبتلا.  كما ويمكن رؤية نمو فطري ابيض علي السطح السفلي للورقة. ومع تقدم المرض، تجف الورقة باكملها بعد تحولها للون البني ثم تموت. كما وتنتشر نفس البقع علي الساق و اعناق الاوراق. كما وتظهر بقع زرقاء رماديه علي قشور البطاطس ومن ثم يتحول اللب الي اللون البني مما يجعلها غير صالحة للاكل. تعفن الحقول المصابة يعطي رائحة مميزة.
<hr>
                        اجراءات وقائية
                        1- عليك استخدام تقاوي سليمة و اصناف مقاومة للمرض. 
                        2- وتأكد ايضاً من وجود نظام تهويه وصرف جيدين. 
                        3- كما ويجب متابعه الزراعة والتخلص من النباتات المصابة وما حولها. 
                        4- ويجب أيضا إتباع التناوب الزراعي لمدة 2الي 3سنوات مع نباتات غير مضيفه للمرض. 
                        5- كما ويجب التخلص من المضيفات البديلة والنباتات الناتجة من نمو بقايا المحصول السابق من الحقل وما حوله. 
                        6- وعدم المبالغة في استخدام التسميد النيتروجينى. 
                        7- استخدام مقويات النباتات. 
                        8- كما ويجب تخزين البطاطس في درجات حرارة منخفضة و تهوية جيدة. 
                        9- والتخلص من بقايا المحصول السابق بعد الحصاد بدفنها عميقا تحت سطح التربة (اكثر من 60سم) او إطعامها الحيوانات.
                        المكافحة الكيميائية
                        1-مبيدات الفطريات hexaconazole 5.0 WP, captan 70.0 WP
                        2-مبيدات الفطريات mancozeb 50.0 WS
                        2-مبيدات الفطريات propineb 70.0 WP
                        
                        سبب المرض
                        المسبب المرضي هو فطر إجباري التطفل. وهذا يعني ضرورة بقائة متصلا ببقايا النباتات و الدرنات او المضيفات البديلة. يمكن للفطر الدخول عبر الجروح والتشققات في سطح النبات. تستطيع الجراثيم النمو في درجات حرارة اعلي خلال فصل الربيع وتنتقل عبر الرياح و المياة. تشتد الاصابة في الاوقات ذات الليالي الباردة (اقل من 18مئوي)  والنهار الدافئ (بين 18و22مئوي) وظروف الرطوبة مثل الامطار والضباب (رطوبة نسبية 90%). في مثل تلك الظروف. يمكن ان تصبح آثار مرض اللفحة المتأخرة مدمرة.
                        `
                        
                    } if (data[class_idx] == "Tomato___Target_Spot") {
                        document.querySelector('.print').innerHTML =`<h2>نبات الطماطم مصاب بالتبقع </h2>`
                        document.querySelector('.read').innerHTML =`نبات الطماطم مصاب بالتبقع`
                    } if (data[class_idx] == "Tomato___Tomato_mosaic_virus") {
                        document.querySelector('.print').innerHTML =`<h2>نبات الطاطم مصاب بفيروس فسيفساء</h2>`
                        document.querySelector('.read').innerHTML =`نبات الطاطم مصاب بفيروس فسيفساء`
                    } if (data[class_idx] == "Tomato___Tomato_Yellow_Leaf_Curl_Virus") {
                        document.querySelector('.print').innerHTML =`<h2>نبات الطماطم مصاب بتجعد الأوراق الصفراء</h2>`
                        document.querySelector('.read').innerHTML =`نبات الطماطم مصاب بتجعد الأوراق الصفراء`
                    
                    } if (data[class_idx] == "healthy") {
                        document.querySelector('.print').innerHTML =`<h2>نباتك غير مصاب (صحي)</h2>
                        <h2>نصائح للحفاظ علي صحة نباتك 🌿</h2>
                        يجب عليك شراء المواد الزراعية من مصادر معتمده. فحص التقاوي والشتلات بعناية قبل شرائها. السماح بمساحه تباعد كافية بين المحاصيل النباتية للسماح بالتهوية الجيدة. اختر الموقع (التربة والطقس) بعناية وتأكد من عدم زراعة أصناف حساسة . التسميد بخليط الأسمدة المناسب وإمداده بالمغذيات والتسميد المتوازن. وتجنب الإفراط في الري أو الإفراط في التسميد. لا تلمس النباتات الصحية بعد لمس النباتات المصابة. تجنب التغيرات الحادة في درجات الحرارة. الحفاظ علي عدد كبير من الأنواع مختلفة من النباتات حول الحقول. إذا كان العلاج ضد مرض معدي ويسهل انتشاره، استخدام منتجات محددة لا تؤثر علي الحشرات المفيدة. وتذكر إزالة الأوراق، الثمار أو الفروع المريضة في الوقت المناسب خلال موسم النمو. في الخريف، نظف بقايا النباتات من الحقل أو البستان وقم بحرقها.
                        `
                        document.querySelector('.read').innerHTML =`نباتك غير مصاب (صحي)
                        نصائح للحفاظ علي صحة نباتك 
                        يجب عليك شراء المواد الزراعية من مصادر معتمده. فحص التقاوي والشتلات بعناية قبل شرائها. السماح بمساحه تباعد كافية بين المحاصيل النباتية للسماح بالتهوية الجيدة. اختر الموقع (التربة والطقس) بعناية وتأكد من عدم زراعة أصناف حساسة . التسميد بخليط الأسمدة المناسب وإمداده بالمغذيات والتسميد المتوازن. وتجنب الإفراط في الري أو الإفراط في التسميد. لا تلمس النباتات الصحية بعد لمس النباتات المصابة. تجنب التغيرات الحادة في درجات الحرارة. الحفاظ علي عدد كبير من الأنواع مختلفة من النباتات حول الحقول. إذا كان العلاج ضد مرض معدي ويسهل انتشاره، استخدام منتجات محددة لا تؤثر علي الحشرات المفيدة. وتذكر إزالة الأوراق، الثمار أو الفروع المريضة في الوقت المناسب خلال موسم النمو. في الخريف، نظف بقايا النباتات من الحقل أو البستان وقم بحرقها.
                        `
                    } if (data[class_idx] == "not_a_plant") {
                        document.querySelector('.print').innerHTML =`<h2>هذه ليست صورة نبات</h2>
                        <h2>الرجاء التقاط صوره للنبات كما موضح في صور المثال </h2>
                        
                        <section class="steps section container">
                        <div class="steps__bg">
                            <h3 class="section__title-center steps__title">
                                أمثلة للصور المطلوبة:
                            </h3>
        
                            <div class="steps__container grid">
                                <div class="steps__card">
                                    <img src="assets/img/TomatoYellowCurlVirus3.jpg" alt="" class="product__img">
        
                                </div>
        
                                <div class="steps__card">
                                    <img src="assets/img/TomatoYellowCurlVirus1.jpg" alt="" class="product__img">
        
                                </div>
        
                                <div class="steps__card">
                                    <img src="assets/img/PotatoHealthy1.jpg" alt="" class="product__img">
        
                                </div>
                            </div>
                        </div>
                        </section>
                
                        `
                        document.querySelector('.read').innerHTML =`هذه ليست صورة نبات
                        الرجاء التقاط صوره للنبات كما موضح في صور المثال
                        أمثلة للصور المطلوبة:
                        `
                    }
                }
            );
            
        }

        

        fileUpload.addEventListener('change', function(e){
            
            let uploadedImage = e.target.value
            if (uploadedImage){
                document.getElementById("blankFile-1").innerHTML = uploadedImage.replace("C:\\fakepath\\","")
                document.getElementById("choose-text-1").innerText = "تغيير الصورة المختارة"
                document.querySelector(".success-1").style.display = "inline-block"

                let extension = uploadedImage.split(".")[1]
                if (!(["doc","docx","pdf"].includes(extension))){
                    document.querySelector(".success-1 i").style.border = "1px solid limegreen"
                    document.querySelector(".success-1 i").style.color = "limegreen"
                }else{
                    document.querySelector(".success-1 i").style.border = "1px solid rgb(25,110,180)"
                    document.querySelector(".success-1 i").style.color = "rgb(25,110,180)"
                }
            }
            let file = this.files[0]
            if (file){
                boxResult.style.display = 'block'
                const reader = new FileReader();
                reader.readAsDataURL(file);
                reader.addEventListener("load", function(){
                    
                    img.style.display = "block"
                    img.setAttribute('src', this.result);
                });
            }
//{"0": "Apple___Apple_scab", "1": "Apple___Black_rot", "2": "Apple___Cedar_apple_rust", "3": "Apple___healthy", "4": "Blueberry___healthy", "5": "Cherry_(including_sour)___Powdery_mildew", "6": "Cherry_(including_sour)___healthy", "7": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "8": "Corn_(maize)___Common_rust_", "9": "Corn_(maize)___Northern_Leaf_Blight", "10": "Corn_(maize)___healthy", "11": "Grape___Black_rot", "12": "Grape___Esca_(Black_Measles)", "13": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "14": "Grape___healthy", "15": "Orange___Haunglongbing_(Citrus_greening)", "16": "Peach___Bacterial_spot", "17": "Peach___healthy", "18": "Pepper,_bell___Bacterial_spot", "19": "Pepper,_bell___healthy", "20": "Potato___Early_blight", "21": "Potato___Late_blight", "22": "Potato___healthy", "23": "Raspberry___healthy", "24": "Soybean___healthy", "25": "Squash___Powdery_mildew", "26": "Strawberry___Leaf_scorch", "27": "Strawberry___healthy", "28": "Tomato___Bacterial_spot", "29": "Tomato___Early_blight", "30": "Tomato___Late_blight", "31": "Tomato___Leaf_Mold", "32": "Tomato___Septoria_leaf_spot", "33": "Tomato___Spider_mites Two-spotted_spider_mite", "34": "Tomato___Target_Spot", "35": "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "36": "Tomato___Tomato_mosaic_virus", "37": "Tomato___healthy"}
            else{
            img.setAttribute("src", "");
            }

            initialize().then( () => { 
                predict()
            })
        })