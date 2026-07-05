class MazeDataLoader extends Object
{
   function MazeDataLoader()
   {
      super();
      this.loader = new LoadVars();
   }
   function loadData(query, data, dataIndex)
   {
      this.loader.onLoad = function(success)
      {
         if(success)
         {
            data[dataIndex] = MazeDataLoader.decodeMessage(this.r);
         }
      };
      this.loader.load(query);
   }
   static function decodeMessage(m)
   {
      var _loc4_ = {};
      var _loc3_ = Base64.Decode(m).split("&");
      var _loc1_ = 0;
      while(_loc1_ < _loc3_.length)
      {
         var _loc2_ = _loc3_[_loc1_].split("=");
         _loc4_[_loc2_[0]] = _loc2_[1];
         _loc1_ = _loc1_ + 1;
      }
      return _loc4_;
   }
}
