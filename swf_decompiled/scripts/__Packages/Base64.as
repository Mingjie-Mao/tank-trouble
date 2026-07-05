class Base64 extends Object
{
   static var _EndOfInput = -1;
   static var _Chars = new Array("A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","0","1","2","3","4","5","6","7","8","9","+","/");
   static var _CharsReverseLookupInited = Base64.InitReverseChars();
   static var _Digits = new Array("0","1","2","3","4","5","6","7","8","9","a","b","c","d","e","f");
   function Base64()
   {
      super();
   }
   static function Encode(str)
   {
      var _loc1_ = new Base64();
      return _loc1_.encodeBase64(str);
   }
   static function Decode(str)
   {
      var _loc1_ = new Base64();
      return _loc1_.decodeBase64(str);
   }
   static function StringReplaceAll(source, find, replacement)
   {
      return source.split(find).join(replacement);
   }
   static function InitReverseChars()
   {
      Base64._CharsReverseLookup = new Array();
      var _loc1_ = 0;
      while(_loc1_ < Base64._Chars.length)
      {
         Base64._CharsReverseLookup[Base64._Chars[_loc1_]] = _loc1_;
         _loc1_ = _loc1_ + 1;
      }
      return true;
   }
   static function UrlDecode(str)
   {
      str = Base64.StringReplaceAll(str,"\\"," ");
      str = unescape(str);
      return str;
   }
   static function UrlEncode(str)
   {
      str = escape(str);
      str = Base64.StringReplaceAll(str,"\\","%2B");
      str = Base64.StringReplaceAll(str,"%20","+");
      return str;
   }
   function setBase64Str(str)
   {
      this._base64Str = str;
      this._base64Count = 0;
   }
   function readBase64()
   {
      if(!this._base64Str)
      {
         return Base64._EndOfInput;
      }
      if(this._base64Count >= this._base64Str.length)
      {
         return Base64._EndOfInput;
      }
      var _loc2_ = this._base64Str.charCodeAt(this._base64Count) & 255;
      this._base64Count = this._base64Count + 1;
      return _loc2_;
   }
   function encodeBase64(str)
   {
      this.setBase64Str(str);
      var _loc3_ = "";
      var _loc2_ = new Array(3);
      var _loc5_ = 0;
      var _loc4_ = false;
      while(!_loc4_ && (_loc2_[0] = this.readBase64()) != Base64._EndOfInput)
      {
         _loc2_[1] = this.readBase64();
         _loc2_[2] = this.readBase64();
         _loc3_ += Base64._Chars[_loc2_[0] >> 2];
         if(_loc2_[1] != Base64._EndOfInput)
         {
            _loc3_ += Base64._Chars[_loc2_[0] << 4 & 48 | _loc2_[1] >> 4];
            if(_loc2_[2] != Base64._EndOfInput)
            {
               _loc3_ += Base64._Chars[_loc2_[1] << 2 & 60 | _loc2_[2] >> 6];
               _loc3_ += Base64._Chars[_loc2_[2] & 63];
            }
            else
            {
               _loc3_ += Base64._Chars[_loc2_[1] << 2 & 60];
               _loc3_ += "=";
               _loc4_ = true;
            }
         }
         else
         {
            _loc3_ += Base64._Chars[_loc2_[0] << 4 & 48];
            _loc3_ += "=";
            _loc3_ += "=";
            _loc4_ = true;
         }
         _loc5_ += 4;
         if(_loc5_ >= 76)
         {
            _loc3_ += "\n";
            _loc5_ = 0;
         }
      }
      return _loc3_;
   }
   function readReverseBase64()
   {
      if(!this._base64Str)
      {
         return Base64._EndOfInput;
      }
      while(true)
      {
         if(this._base64Count >= this._base64Str.length)
         {
            return Base64._EndOfInput;
         }
         var _loc2_ = this._base64Str.charAt(this._base64Count);
         this._base64Count = this._base64Count + 1;
         if(Base64._CharsReverseLookup[_loc2_])
         {
            return Base64._CharsReverseLookup[_loc2_];
         }
         if(_loc2_ == "A")
         {
            return 0;
         }
      }
      return Base64._EndOfInput;
   }
   function ntos(n)
   {
      var _loc1_ = n.toString(16);
      if(_loc1_.length == 1)
      {
         _loc1_ = "0" + _loc1_;
      }
      _loc1_ = "%" + _loc1_;
      return unescape(_loc1_);
   }
   function decodeBase64(str)
   {
      this.setBase64Str(str);
      var _loc3_ = "";
      var _loc2_ = new Array(4);
      var _loc4_ = false;
      while(!_loc4_ && (_loc2_[0] = this.readReverseBase64()) != Base64._EndOfInput && (_loc2_[1] = this.readReverseBase64()) != Base64._EndOfInput)
      {
         _loc2_[2] = this.readReverseBase64();
         _loc2_[3] = this.readReverseBase64();
         _loc3_ += this.ntos(_loc2_[0] << 2 & 255 | _loc2_[1] >> 4);
         if(_loc2_[2] != Base64._EndOfInput)
         {
            _loc3_ += this.ntos(_loc2_[1] << 4 & 255 | _loc2_[2] >> 2);
            if(_loc2_[3] != Base64._EndOfInput)
            {
               _loc3_ += this.ntos(_loc2_[2] << 6 & 255 | _loc2_[3]);
            }
            else
            {
               _loc4_ = true;
            }
         }
         else
         {
            _loc4_ = true;
         }
      }
      return _loc3_;
   }
   function toHex(n)
   {
      var _loc4_ = "";
      var _loc3_ = true;
      var _loc1_ = 32;
      while(_loc1_ > 0)
      {
         _loc1_ -= 4;
         var _loc2_ = n >> _loc1_ & 15;
         if(!_loc3_ || _loc2_ != 0)
         {
            _loc3_ = false;
            _loc4_ += Base64._Digits[_loc2_];
         }
      }
      return _loc4_ != "" ? _loc4_ : "0";
   }
   function pad(str, len, pad)
   {
      var _loc2_ = str;
      var _loc1_ = str.length;
      while(_loc1_ < len)
      {
         _loc2_ = pad + _loc2_;
         _loc1_ = _loc1_ + 1;
      }
      return _loc2_;
   }
   function encodeHex(str)
   {
      var _loc4_ = "";
      var _loc2_ = 0;
      while(_loc2_ < str.length)
      {
         _loc4_ += this.pad(this.toHex(str.charCodeAt(_loc2_) & 255),2,"0");
         _loc2_ = _loc2_ + 1;
      }
      return _loc4_;
   }
   function decodeHex(str)
   {
      var _loc5_ = "";
      var _loc3_ = "";
      var _loc2_ = 0;
      while(_loc2_ < str.length)
      {
         _loc3_ += str.charAt(_loc2_);
         if(_loc3_.length == 2)
         {
            _loc5_ += this.ntos(parseInt("0x" + _loc3_));
            _loc3_ = "";
         }
         _loc2_ = _loc2_ + 1;
      }
      return _loc5_;
   }
}
