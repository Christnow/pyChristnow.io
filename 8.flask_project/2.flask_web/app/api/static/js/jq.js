$(function() {
	$('#yes').click(function(event) {
		modal('我就知道你一定会愿意的。(^_^)', function() {
			$('.page_one').addClass('hide');
			$('.page_two').removeClass('hide');
			typeWrite();
			fireworks();
			
		});
	});
	$('#no').click(function(event) {
		modal('明人不说暗话！', A);
	});
});

function A() {
	modal('我喜欢你！', B);
}

function B() {
	modal('我知道你在等我这一句话', C);
}

function C() {
	modal('请您不要拒绝我', D);
}

function D() {
	modal('拒绝我，不存在的', E);
}

function E() {
	modal('这辈子都不可能让你离开我', F);
}

function F() {
	modal('跟我走吧', G);
}

function G() {
	modal('房产证上写你名', H);
}

function H() {
	modal('我会做饭', I);
}

function I() {
	modal('我会做家务', J)
}

function J() {
	modal('答应我吧', function() {
//		fireworks();
	});
}

function fireworks() {
	$('.page_one').addClass('hide');
	$('.page_two').removeClass('hide');
	$('.page_two').fireworks({ 
	  sound: false, 
	  opacity: 0.9, 
	  width: '100%', 
	  height: '100%' 
	});	
}

function modal(content, callback) {
	var tpl = '<div class="container">'+
							'<div class="mask"></div>'+
							'<div class="modal">'+
								'<p>'+ content +'</p>'+
								'<button type="button" id="confirm" class="confirm">确定</button>'+
							'</div>'+
						'</div>';
	$('body').append(tpl);
	$(document).on('click', '.confirm', function() {
		$('.container').remove();
		callback();
	});
}

var myWords = '有人说，人的一生会遇到2920万人，而两个人相爱的概率只有0.000049。在这茫茫人海中，两个人能相遇就值得感激，能相爱更是一种难得。所以，我很庆幸上天让我遇见了你。我希望有个如你一般的人，能看完我写过的所有状态，读完我所有的日志，看完我从小到大的照片，试着听我喜欢的歌。如果可以，甚至陪我去我喜欢的地方，只想弥补错过你的青春。——许肖飞飞';
var x = 0;
var speed = 150;
var current = 0;
function typeWrite(){
	$('.type_words').html(myWords.substring(0, x++)+'_');
	var timer = setTimeout("typeWrite()", speed);
	if (x == myWords.length) {
		x = myWords.length;
		clearTimeout(timer)
	}
}
