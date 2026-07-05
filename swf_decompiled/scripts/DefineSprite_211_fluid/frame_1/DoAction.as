function addVolume(pos, size, initialVel)
{
   var _loc11_ = Math.pow(m / rho,0.5);
   var _loc9_ = _loc11_ * 0.8700000000000001;
   var _loc7_ = pos.x - size;
   while(_loc7_ < pos.x + size)
   {
      var _loc5_ = pos.y - size;
      while(_loc5_ < pos.y + size)
      {
         particles.push({x:_loc7_ + Math.random() - 0.5,y:_loc5_ + Math.random() - 0.5,vel:initialVel,force:{x:0,y:0},pressure:0,density:0});
         _loc5_ += _loc9_;
      }
      _loc7_ += _loc9_;
   }
}
function SPH_ComputePressure()
{
   var _loc2_ = 0;
   while(_loc2_ < particles.length)
   {
      var _loc7_ = hSquared;
      var _loc1_ = 0;
      while(_loc1_ < particles.length)
      {
         if(_loc2_ != _loc1_)
         {
            var _loc6_ = particles[_loc1_].x - particles[_loc2_].x;
            var _loc5_ = particles[_loc1_].y - particles[_loc2_].y;
            var _loc4_ = _loc6_ * _loc6_ + _loc5_ * _loc5_;
            if(hSquared > _loc4_)
            {
               var _loc3_ = hSquared - _loc4_;
               _loc7_ += _loc3_ * _loc3_ * _loc3_;
            }
         }
         _loc1_ = _loc1_ + 1;
      }
      particles[_loc2_].density = _loc7_ * m * W_poly6_kernel;
      particles[_loc2_].pressure = k * (particles[_loc2_].density - rho);
      _loc2_ = _loc2_ + 1;
   }
}
function SPH_ComputeForces()
{
   var _loc1_ = 0;
   while(_loc1_ < particles.length)
   {
      var _loc12_ = 0;
      particles[_loc1_].force.x = 0;
      particles[_loc1_].force.y = 0;
      var _loc2_ = 0;
      while(_loc2_ < particles.length)
      {
         if(_loc1_ != _loc2_)
         {
            var _loc6_ = particles[_loc2_].x - particles[_loc1_].x;
            var _loc5_ = particles[_loc2_].y - particles[_loc1_].y;
            var _loc7_ = _loc6_ * _loc6_ + _loc5_ * _loc5_;
            if(hSquared > _loc7_)
            {
               var _loc4_ = Math.sqrt(_loc7_);
               var _loc3_ = h - _loc4_;
               var _loc11_ = _loc6_ / _loc4_;
               var _loc10_ = _loc5_ / _loc4_;
               _loc11_ *= -0.5 * W_spiky_gradient_kernel * _loc3_ * _loc3_ * (particles[_loc1_].pressure + particles[_loc2_].pressure) * (m / particles[_loc1_].density) * (m / particles[_loc2_].density) / _loc4_;
               _loc10_ *= -0.5 * W_spiky_gradient_kernel * _loc3_ * _loc3_ * (particles[_loc1_].pressure + particles[_loc2_].pressure) * (m / particles[_loc1_].density) * (m / particles[_loc2_].density) / _loc4_;
               particles[_loc1_].force.x += _loc11_;
               particles[_loc1_].force.y += _loc10_;
               var _loc9_ = particles[_loc2_].vel.x - particles[_loc1_].vel.x;
               var _loc8_ = particles[_loc2_].vel.y - particles[_loc1_].vel.y;
               _loc9_ *= 0.5 * W_viscosity_laplacian_kernel * _loc3_ * nu * (m / particles[_loc1_].density) * (m / particles[_loc2_].density);
               _loc8_ *= 0.5 * W_viscosity_laplacian_kernel * _loc3_ * nu * (m / particles[_loc1_].density) * (m / particles[_loc2_].density);
               particles[_loc1_].force.x += _loc9_;
               particles[_loc1_].force.y += _loc8_;
            }
         }
         _loc2_ = _loc2_ + 1;
      }
      _loc1_ = _loc1_ + 1;
   }
}
function integrate(dt)
{
   var _loc1_ = 0;
   while(_loc1_ < particles.length)
   {
      particles[_loc1_].x += particles[_loc1_].vel.x * dt;
      particles[_loc1_].y += particles[_loc1_].vel.y * dt;
      particles[_loc1_].vel.x += particles[_loc1_].force.x / m * dt * dt;
      particles[_loc1_].vel.y += particles[_loc1_].force.y / m * dt * dt;
      _loc1_ = _loc1_ + 1;
   }
}
function draw()
{
   this.clear();
   this.lineStyle(3,5592575);
   var _loc2_ = 0;
   while(_loc2_ < particles.length)
   {
      this.moveTo(particles[_loc2_].x,particles[_loc2_].y);
      this.lineTo(particles[_loc2_].x + 1,particles[_loc2_].y);
      _loc2_ = _loc2_ + 1;
   }
}
var particles = new Array();
var h = 2.5;
var hSquared = h * h;
var r = 1;
var m = 1;
var k = 1;
var rho = 0.1;
var nu = 100;
var W_poly6_kernel = 4 / (3.141592653589793 * h * h);
var W_spiky_gradient_kernel = -30 / (3.141592653589793 * h * h);
var W_viscosity_laplacian_kernel = 60 / (34.55751918948772 * h * h);
addVolume({x:50,y:45},4,{x:0,y:1});
addVolume({x:53,y:65},4,{x:0,y:-1});
onEnterFrame = function()
{
   if(_root.frozen)
   {
      return undefined;
   }
   var _loc2_ = 0;
   while(_loc2_ < 40)
   {
      SPH_ComputePressure();
      SPH_ComputeForces();
      integrate(0.05);
      _loc2_ = _loc2_ + 1;
   }
   draw();
};
